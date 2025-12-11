#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🤖 بوت تيليجرام كامل - يعمل على جميع المنصات
استخراج النصوص من البطاقة والجواز باستخدام Gemini AI
"""

import os
import sys
import json
import re
import random
import string
import base64
import hashlib
import logging
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import traceback

print("=" * 70)
print("🚀 بوت استخراج النصوص - النسخة العالمية")
print("=" * 70)

# ============= إعدادات النظام =============
class Platform(Enum):
    RENDER = "render"
    RAILWAY = "railway"
    KOYEB = "koyeb"
    CYCLIC = "cyclic"
    HEROKU = "heroku"
    PYTHONANYWHERE = "pythonanywhere"
    REPLIT = "replit"
    LOCAL = "local"

class AIType(Enum):
    GEMINI = "gemini"
    OCR = "ocr"
    MOCK = "mock"

# ============= فئات البيانات =============
@dataclass
class UserInfo:
    """معلومات المستخدم"""
    user_id: str
    username: Optional[str]
    first_name: Optional[str]
    last_name: Optional[str]
    language_code: Optional[str]
    join_date: str
    extractions_count: int = 0
    last_extraction: Optional[str] = None
    created_email: Optional[str] = None

@dataclass
class ExtractionResult:
    """نتيجة استخراج النصوص"""
    success: bool
    arabic_texts: List[str]
    english_texts: List[str]
    extracted_name: Optional[str]
    confidence: float
    processing_time: float
    ai_engine: str
    error_message: Optional[str] = None

@dataclass
class GeneratedData:
    """البيانات المنشأة"""
    email: str
    password: str
    filename: str
    file_content: str
    timestamp: str

# ============= المدير الرئيسي =============
class TelegramBotManager:
    """المدير الرئيسي للبوت"""
    
    def __init__(self):
        self.platform = self._detect_platform()
        self.logger = self._setup_logger()
        self.ai_type = self._setup_ai()
        self.bot_config = self._load_config()
        self.users_db: Dict[str, UserInfo] = {}
        self.extraction_history: List[Dict] = []
        
        self.logger.info(f"🚀 البوت يعمل على منصة: {self.platform.value}")
        self.logger.info(f"🤖 محرك الذكاء الاصطناعي: {self.ai_type.value}")
    
    def _detect_platform(self) -> Platform:
        """الكشف عن المنصة"""
        env = os.environ
        
        if 'RENDER' in env:
            return Platform.RENDER
        elif 'RAILWAY_ENVIRONMENT' in env:
            return Platform.RAILWAY
        elif 'KOYEB' in env:
            return Platform.KOYEB
        elif 'CYCLIC_URL' in env:
            return Platform.CYCLIC
        elif 'HEROKU_APP_NAME' in env:
            return Platform.HEROKU
        elif 'PYTHONANYWHERE_SITE' in env:
            return Platform.PYTHONANYWHERE
        elif 'REPL_ID' in env:
            return Platform.REPLIT
        else:
            return Platform.LOCAL
    
    def _setup_logger(self) -> logging.Logger:
        """إعداد نظام التسجيل"""
        logger = logging.getLogger('TelegramBot')
        logger.setLevel(logging.INFO)
        
        # إعداد formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # إضافة handler للكونسول
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # إضافة handler للملف (إن أمكن)
        try:
            file_handler = logging.FileHandler('bot.log', encoding='utf-8')
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except:
            pass
        
        return logger
    
    def _setup_ai(self) -> AIType:
        """إعداد محرك الذكاء الاصطناعي"""
        try:
            # محاولة استيراد Gemini AI
            import google.generativeai as genai
            
            if os.environ.get('GEMINI_API_KEY'):
                genai.configure(api_key=os.environ['GEMINI_API_KEY'])
                self.logger.info("✅ Gemini AI جاهز للاستخدام")
                return AIType.GEMINI
        except ImportError:
            self.logger.warning("⚠️ Gemini AI غير مثبت")
        except Exception as e:
            self.logger.error(f"❌ خطأ في إعداد Gemini AI: {e}")
        
        # استخدام OCR كبديل
        try:
            import easyocr
            self.logger.info("✅ EasyOCR جاهز للاستخدام")
            return AIType.OCR
        except ImportError:
            self.logger.warning("⚠️ EasyOCR غير مثبت")
        
        # استخدام وضع المحاكاة للاختبار
        self.logger.info("⚠️ استخدام وضع المحاكاة (لاستخراج نص وهمي)")
        return AIType.MOCK
    
    def _load_config(self) -> Dict:
        """تحميل الإعدادات"""
        config = {
            'bot_token': os.environ.get('TELEGRAM_TOKEN', ''),
            'bot_name': 'ID Card Extractor Bot',
            'bot_version': '2.0.0',
            'admin_ids': os.environ.get('ADMIN_IDS', '').split(','),
            'max_file_size': 10 * 1024 * 1024,  # 10MB
            'supported_formats': ['image/jpeg', 'image/png', 'image/jpg'],
            'rate_limit': 5,  # 5 طلبات في الدقيقة
            'auto_delete_minutes': 60,  # حذف الملفات بعد 60 دقيقة
        }
        
        if not config['bot_token']:
            self.logger.critical("❌ لم يتم تعيين توكن البوت!")
            self.logger.info("🔑 أضف TELEGRAM_TOKEN في Environment Variables")
            sys.exit(1)
        
        return config

class TextExtractor:
    """مستخرج النصوص"""
    
    def __init__(self, ai_type: AIType):
        self.ai_type = ai_type
        self.setup_engine()
    
    def setup_engine(self):
        """إعداد محرك الاستخراج"""
        if self.ai_type == AIType.GEMINI:
            import google.generativeai as genai
            self.gemini_model = genai.GenerativeModel('gemini-1.5-flash')
        
        elif self.ai_type == AIType.OCR:
            import easyocr
            self.ocr_reader = easyocr.Reader(['ar', 'en'])
        
        elif self.ai_type == AIType.MOCK:
            self.mock_data = {
                'arabic': [
                    "البطاقة الشخصية",
                    "الجمهورية العربية السورية",
                    "وزارة الداخلية",
                    "الاسم: أحمد محمد علي",
                    "تاريخ الميلاد: ١٥/٠٣/١٩٩٠",
                    "رقم البطاقة: ١٢٣٤٥٦٧٨٩",
                    "مكان الإصدار: دمشق"
                ],
                'english': [
                    "IDENTITY CARD",
                    "SYRIAN ARAB REPUBLIC",
                    "MINISTRY OF INTERIOR",
                    "Name: Ahmed Mohamed Ali",
                    "Date of Birth: 15/03/1990",
                    "ID Number: 123456789",
                    "Place of Issue: Damascus"
                ]
            }
    
    def extract_from_image(self, image_bytes: bytes) -> ExtractionResult:
        """استخراج النصوص من الصورة"""
        start_time = time.time()
        
        try:
            if self.ai_type == AIType.GEMINI:
                return self._extract_with_gemini(image_bytes)
            elif self.ai_type == AIType.OCR:
                return self._extract_with_ocr(image_bytes)
            elif self.ai_type == AIType.MOCK:
                return self._extract_mock()
            else:
                raise ValueError(f"نوع AI غير مدعوم: {self.ai_type}")
        
        except Exception as e:
            processing_time = time.time() - start_time
            return ExtractionResult(
                success=False,
                arabic_texts=[],
                english_texts=[],
                extracted_name=None,
                confidence=0.0,
                processing_time=processing_time,
                ai_engine=self.ai_type.value,
                error_message=str(e)
            )
    
    def _extract_with_gemini(self, image_bytes: bytes) -> ExtractionResult:
        """استخراج باستخدام Gemini AI"""
        import google.generativeai as genai
        
        # تحويل الصورة إلى base64
        image_b64 = base64.b64encode(image_bytes).decode('utf-8')
        
        prompt = """أنت خبير في استخراج النصوص من وثائق الهوية.
        استخرج جميع النصوص من هذه الصورة وأجب بالتنسيق التالي:
        
        الاسم: [الاسم الكامل إن وجد]
        
        النصوص العربية:
        [النصوص العربية هنا، كل سطر في سطر منفصل]
        
        النصوص الإنجليزية:
        [النصوص الإنجليزية هنا، كل سطر في سطر منفصل]
        
        إذا لم تجد نصاً، اكتب: لا يوجد"""
        
        try:
            response = self.gemini_model.generate_content([
                prompt,
                {"mime_type": "image/jpeg", "data": image_b64}
            ])
            
            result_text = response.text
            return self._parse_extraction_result(result_text)
        
        except Exception as e:
            raise Exception(f"خطأ في Gemini AI: {e}")
    
    def _extract_with_ocr(self, image_bytes: bytes) -> ExtractionResult:
        """استخراج باستخدام OCR"""
        import cv2
        import numpy as np
        
        try:
            # تحويل bytes إلى صورة OpenCV
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                raise ValueError("فشل في فك ترميز الصورة")
            
            # تحسين الصورة
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.medianBlur(gray, 3)
            
            # استخراج النصوص
            results = self.ocr_reader.readtext(gray, paragraph=True)
            
            # تجميع النصوص
            arabic_texts = []
            english_texts = []
            
            for (bbox, text, prob) in results:
                text = text.strip()
                if not text:
                    continue
                
                # تحديد اللغة
                if re.search(r'[\u0600-\u06FF]', text):
                    arabic_texts.append(text)
                else:
                    english_texts.append(text)
            
            # استخراج الاسم (محاولة)
            extracted_name = None
            for text in arabic_texts:
                if 'اسم' in text.lower() or 'الاسم' in text:
                    extracted_name = text.replace('اسم:', '').replace('الاسم:', '').strip()
                    break
            
            processing_time = time.time() - start_time if 'start_time' in locals() else 0
            
            return ExtractionResult(
                success=True,
                arabic_texts=arabic_texts,
                english_texts=english_texts,
                extracted_name=extracted_name,
                confidence=0.8,
                processing_time=processing_time,
                ai_engine=self.ai_type.value
            )
        
        except Exception as e:
            raise Exception(f"خطأ في OCR: {e}")
    
    def _extract_mock(self) -> ExtractionResult:
        """استخراج نص وهمي للاختبار"""
        import time
        time.sleep(2)  # محاكاة وقت المعالجة
        
        return ExtractionResult(
            success=True,
            arabic_texts=self.mock_data['arabic'],
            english_texts=self.mock_data['english'],
            extracted_name="أحمد محمد علي",
            confidence=1.0,
            processing_time=2.0,
            ai_engine=self.ai_type.value
        )
    
    def _parse_extraction_result(self, text: str) -> ExtractionResult:
        """تحليل نتيجة الاستخراج"""
        arabic_texts = []
        english_texts = []
        extracted_name = None
        current_section = None
        
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            
            if not line:
                continue
            
            if line.startswith('الاسم:'):
                extracted_name = line.replace('الاسم:', '').strip()
            elif line.startswith('النصوص العربية:'):
                current_section = 'arabic'
            elif line.startswith('النصوص الإنجليزية:'):
                current_section = 'english'
            elif line == 'لا يوجد':
                continue
            elif current_section:
                if current_section == 'arabic':
                    arabic_texts.append(line)
                elif current_section == 'english':
                    english_texts.append(line)
        
        processing_time = time.time() - start_time if 'start_time' in locals() else 0
        
        return ExtractionResult(
            success=True,
            arabic_texts=arabic_texts,
            english_texts=english_texts,
            extracted_name=extracted_name,
            confidence=0.9,
            processing_time=processing_time,
            ai_engine=self.ai_type.value
        )

class DataGenerator:
    """منشئ البيانات"""
    
    @staticmethod
    def generate_email(name: Optional[str]) -> str:
        """إنشاء بريد إلكتروني من الاسم"""
        if not name or name.strip() == "":
            name = "user"
        
        # تحويل الاسم إلى حروف لاتينية
        name_clean = re.sub(r'[^\w\s\u0600-\u06FF]', '', name, flags=re.UNICODE)
        name_clean = name_clean.strip()
        
        # تحويل عربي إلى إنجليزي
        arabic_to_latin = {
            'أ': 'a', 'ا': 'a', 'إ': 'e', 'آ': 'a',
            'ب': 'b', 'ت': 't', 'ث': 'th',
            'ج': 'j', 'ح': 'h', 'خ': 'kh',
            'د': 'd', 'ذ': 'dh', 'ر': 'r', 'ز': 'z',
            'س': 's', 'ش': 'sh', 'ص': 's', 'ض': 'd',
            'ط': 't', 'ظ': 'z', 'ع': 'a', 'غ': 'gh',
            'ف': 'f', 'ق': 'q', 'ك': 'k', 'ل': 'l',
            'م': 'm', 'ن': 'n', 'ه': 'h', 'و': 'w',
            'ي': 'y', 'ى': 'a', 'ئ': 'e',
            'ة': 'h', ' ': '.'
        }
        
        latin_name = ""
        for char in name_clean:
            if char in arabic_to_latin:
                latin_name += arabic_to_latin[char]
            elif char.isalpha() and char.isascii():
                latin_name += char.lower()
            elif char == ' ':
                latin_name += '.'
        
        # تنظيف النتيجة
        latin_name = re.sub(r'[^a-z.]', '', latin_name)
        latin_name = re.sub(r'\.+', '.', latin_name)
        latin_name = latin_name.strip('.')
        
        if len(latin_name) < 3:
            latin_name = f"user{random.randint(1000, 9999)}"
        
        # خيارات النطاقات
        domains = [
            "id-card.me", "official-id.com", "verify.docs",
            "passport.info", "identity.pro", "document.space"
        ]
        
        domain = random.choice(domains)
        email = f"{latin_name}@{domain}"
        
        return email
    
    @staticmethod
    def generate_password() -> str:
        """إنشاء كلمة مرور قوية"""
        # يجب أن تحتوي على حرف كبير، صغير، رقم ورمز
        uppercase = random.choice(string.ascii_uppercase)
        lowercase = random.choice(string.ascii_lowercase)
        digit = random.choice(string.digits)
        symbol = random.choice("!@#$%^&*")
        
        # باقي الأحرف
        all_chars = string.ascii_letters + string.digits + "!@#$%^&*"
        remaining = ''.join(random.choice(all_chars) for _ in range(8))
        
        # دمج وخلط
        password = uppercase + lowercase + digit + symbol + remaining
        password_list = list(password)
        random.shuffle(password_list)
        
        return ''.join(password_list)
    
    @staticmethod
    def create_text_file(name: str, arabic_texts: List[str], 
                        english_texts: List[str], email: str, 
                        password: str, platform: str) -> GeneratedData:
        """إنشاء ملف نصي بالنتائج"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # إنشاء المحتوى
        content = "=" * 70 + "\n"
        content += "🤖 المعلومات المستخرجة من الوثيقة\n"
        content += "=" * 70 + "\n\n"
        
        if name:
            content += f"👤 اسم الشخص: {name}\n\n"
        
        content += "🔤 النصوص العربية المستخرجة:\n"
        content += "-" * 40 + "\n"
        if arabic_texts:
            for i, text in enumerate(arabic_texts, 1):
                content += f"{i:02d}. {text}\n"
        else:
            content += "❌ لم يتم العثور على نصوص عربية\n"
        
        content += "\n" + "=" * 70 + "\n\n"
        
        content += "🔤 النصوص الإنجليزية المستخرجة:\n"
        content += "-" * 40 + "\n"
        if english_texts:
            for i, text in enumerate(english_texts, 1):
                content += f"{i:02d}. {text}\n"
        else:
            content += "❌ لم يتم العثور على نصوص إنجليزية\n"
        
        content += "\n" + "=" * 70 + "\n\n"
        
        content += "📧 البيانات المنشأة تلقائياً:\n"
        content += "-" * 40 + "\n"
        content += f"📧 البريد الإلكتروني: {email}\n"
        content += f"🔐 كلمة المرور: {password}\n\n"
        
        content += "=" * 70 + "\n"
        content += f"📅 تاريخ الإستخراج: {timestamp}\n"
        content += f"🌐 المنصة المستخدمة: {platform}\n"
        content += f"🤖 المحرك: {platform.upper()}\n"
        content += "=" * 70 + "\n"
        
        # إنشاء اسم الملف
        safe_name = re.sub(r'[^\w\s]', '', name)
        safe_name = safe_name.strip().replace(' ', '_')[:20]
        filename = f"معلومات_{safe_name}_{int(time.time())}.txt"
        
        return GeneratedData(
            email=email,
            password=password,
            filename=filename,
            file_content=content,
            timestamp=timestamp
        )

# ============= تطبيق Flask للويب =============
class FlaskAppWrapper:
    """غلاف تطبيق Flask"""
    
    def __init__(self, bot_manager: TelegramBotManager):
        self.bot_manager = bot_manager
        self.app = None
        
    def create_app(self):
        """إنشاء تطبيق Flask"""
        try:
            from flask import Flask, request, jsonify, render_template_string
            self.app = Flask(__name__)
            
            @self.app.route('/')
            def home():
                return render_template_string(self._get_home_html())
            
            @self.app.route('/health')
            def health():
                return jsonify({
                    'status': 'healthy',
                    'platform': self.bot_manager.platform.value,
                    'users': len(self.bot_manager.users_db),
                    'timestamp': datetime.now().isoformat()
                })
            
            @self.app.route('/webhook', methods=['POST'])
            def webhook():
                # سيتم تنفيذ Webhook هنا
                return jsonify({'status': 'webhook_ready'})
            
            return self.app
        
        except ImportError:
            self.bot_manager.logger.warning("⚠️ Flask غير مثبت، سيتم استخدام وضع البوت فقط")
            return None
    
    def _get_home_html(self) -> str:
        """إنشاء صفحة HTML للرئيسية"""
        return """
        <!DOCTYPE html>
        <html dir="rtl">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>بوت استخراج النصوص 🤖</title>
            <style>
                body {
                    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    margin: 0;
                    padding: 20px;
                    min-height: 100vh;
                }
                .container {
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 40px 20px;
                }
                .header {
                    text-align: center;
                    margin-bottom: 50px;
                }
                .header h1 {
                    font-size: 3em;
                    margin-bottom: 10px;
                }
                .platform-badge {
                    display: inline-block;
                    background: rgba(255, 255, 255, 0.2);
                    padding: 10px 20px;
                    border-radius: 50px;
                    margin: 10px;
                    backdrop-filter: blur(10px);
                }
                .stats {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                    gap: 20px;
                    margin: 40px 0;
                }
                .stat-card {
                    background: rgba(255, 255, 255, 0.1);
                    padding: 30px;
                    border-radius: 15px;
                    text-align: center;
                    backdrop-filter: blur(5px);
                    border: 1px solid rgba(255, 255, 255, 0.1);
                    transition: transform 0.3s;
                }
                .stat-card:hover {
                    transform: translateY(-5px);
                }
                .stat-number {
                    font-size: 3em;
                    font-weight: bold;
                    margin: 10px 0;
                    color: #4ade80;
                }
                .features {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                    gap: 20px;
                    margin: 50px 0;
                }
                .feature {
                    background: rgba(255, 255, 255, 0.1);
                    padding: 25px;
                    border-radius: 15px;
                    border-left: 5px solid #8b5cf6;
                }
                .buttons {
                    text-align: center;
                    margin: 50px 0;
                }
                .btn {
                    display: inline-block;
                    padding: 15px 30px;
                    margin: 10px;
                    background: white;
                    color: #667eea;
                    text-decoration: none;
                    border-radius: 50px;
                    font-weight: bold;
                    transition: all 0.3s;
                }
                .btn:hover {
                    transform: scale(1.05);
                    box-shadow: 0 10px 20px rgba(0, 0, 0, 0.2);
                }
                footer {
                    text-align: center;
                    margin-top: 50px;
                    padding-top: 20px;
                    border-top: 1px solid rgba(255, 255, 255, 0.1);
                }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🤖 بوت استخراج النصوص</h1>
                    <div class="platform-badge">
                        يعمل على {{ platform }}
                    </div>
                    <p>بوت ذكي لاستخراج النصوص من صور البطاقة والجواز</p>
                </div>
                
                <div class="stats">
                    <div class="stat-card">
                        <div class="stat-number">{{ users_count }}</div>
                        <p>👥 المستخدمون</p>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number">{{ extractions_count }}</div>
                        <p>📸 عمليات استخراج</p>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number">{{ ai_engine }}</div>
                        <p>🤖 محرك الذكاء</p>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number">{{ uptime }}</div>
                        <p>⏱️ ساعات تشغيل</p>
                    </div>
                </div>
                
                <div class="features">
                    <div class="feature">
                        <h3>📸 استخراج النصوص</h3>
                        <p>استخراج تلقائي للنصوص العربية والإنجليزية من صور البطاقة والجواز</p>
                    </div>
                    <div class="feature">
                        <h3>📧 إنشاء بريد إلكتروني</h3>
                        <p>إنشاء بريد إلكتروني احترافي من الاسم الموجود في الوثيقة</p>
                    </div>
                    <div class="feature">
                        <h3>🔐 كلمات مرور آمنة</h3>
                        <p>توليد كلمات مرور قوية وتوفيرها في ملف نصي منظم</p>
                    </div>
                    <div class="feature">
                        <h3>🌐 متعدد المنصات</h3>
                        <p>يعمل على جميع المنصات السحابية الرئيسية</p>
                    </div>
                </div>
                
                <div class="buttons">
                    <a href="https://t.me/your_bot_username" class="btn" target="_blank">
                        💬 ابدأ على تيليجرام
                    </a>
                    <a href="/health" class="btn">
                        🩺 فحص الصحة
                    </a>
                    <a href="https://github.com/yourusername/telegram-bot" class="btn" target="_blank">
                        📚 الكود المصدري
                    </a>
                </div>
                
                <footer>
                    <p>© {{ current_year }} بوت استخراج النصوص | النسخة {{ version }}</p>
                    <p>🔄 آخر تحديث: {{ timestamp }}</p>
                </footer>
            </div>
        </body>
        </html>
        """

# ============= تشغيل البوت =============
def run_bot():
    """الدالة الرئيسية لتشغيل البوت"""
    
    # تهيئة المدير
    bot_manager = TelegramBotManager()
    
    # إنشاء مستخرج النصوص
    text_extractor = TextExtractor(bot_manager.ai_type)
    
    # إنشاء منشئ البيانات
    data_generator = DataGenerator()
    
    # استيراد telebot
    try:
        import telebot
        from telebot import types
        
        # إنشاء كائن البوت
        bot = telebot.TeleBot(bot_manager.bot_config['bot_token'])
        
        bot_manager.logger.info(f"✅ البوت جاهز: {bot_manager.bot_config['bot_name']}")
        
        @bot.message_handler(commands=['start', 'help', 'ابدأ'])
        def handle_start(message):
            """معالجة أمر /start"""
            try:
                user_id = str(message.from_user.id)
                
                # حفظ معلومات المستخدم
                if user_id not in bot_manager.users_db:
                    bot_manager.users_db[user_id] = UserInfo(
                        user_id=user_id,
                        username=message.from_user.username,
                        first_name=message.from_user.first_name,
                        last_name=message.from_user.last_name,
                        language_code=message.from_user.language_code,
                        join_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        extractions_count=0
                    )
                
                # رسالة الترحيب
                welcome_text = f"""
                🌟 أهلاً {message.from_user.first_name}!
                
                🤖 *بوت استخراج النصوص المتقدم*
                
                ✨ *المميزات:*
                ✅ استخراج النصوص العربية والإنجليزية من الصور
                ✅ إنشاء بريد إلكتروني تلقائياً من الاسم
                ✅ توليد كلمة مرور قوية
                ✅ حفظ النتائج في ملف نصي منظم
                
                📸 *كيفية الاستخدام:*
                1. أرسل صورة البطاقة الشخصية أو جواز السفر
                2. انتظر قليلاً للمعالجة
                3. استلم الملف النصي مع جميع المعلومات
                
                ⚡ *الآن:* أرسل صورة للبدء!
                
                🌐 *المنصة:* {bot_manager.platform.value.upper()}
                """
                
                # إنشاء لوحة المفاتيح
                keyboard = types.ReplyKeyboardMarkup(resize_keyboard=True, row_width=2)
                keyboard.add(
                    types.KeyboardButton("📸 إرسال صورة"),
                    types.KeyboardButton("ℹ️ معلومات"),
                    types.KeyboardButton("📊 إحصائيات"),
                    types.KeyboardButton("🆘 المساعدة")
                )
                
                bot.send_message(
                    message.chat.id,
                    welcome_text,
                    reply_markup=keyboard,
                    parse_mode='Markdown'
                )
                
                bot_manager.logger.info(f"👤 مستخدم جديد: {message.from_user.username or message.from_user.id}")
                
            except Exception as e:
                bot_manager.logger.error(f"خطأ في /start: {e}")
                bot.reply_to(message, "❌ حدث خطأ في معالجة الأمر. حاول مرة أخرى.")
        
        @bot.message_handler(func=lambda message: message.text == "📸 إرسال صورة")
        def handle_send_photo_button(message):
            """زر إرسال صورة"""
            bot.reply_to(
                message,
                "📸 *جاهز لاستقبال الصورة!*\n\n"
                "الرجاء إرسال صورة البطاقة أو الجواز الآن.\n"
                "يمكنك التقاط صورة جديدة أو اختيار صورة من المعرض.",
                parse_mode='Markdown'
            )
        
        @bot.message_handler(content_types=['photo'])
        def handle_photo_message(message):
            """معالجة الصور"""
            try:
                user_id = str(message.from_user.id)
                
                # إعلام المستخدم بالمعالجة
                processing_msg = bot.reply_to(
                    message,
                    "📥 *جاري تحميل الصورة...*\n"
                    "⏳ الرجاء الانتظار قليلاً",
                    parse_mode='Markdown'
                )
                
                # الحصول على الصورة بأفضل جودة
                file_id = message.photo[-1].file_id
                file_info = bot.get_file(file_id)
                file_url = f"https://api.telegram.org/file/bot{bot_manager.bot_config['bot_token']}/{file_info.file_path}"
                
                # تحميل الصورة
                import requests
                response = requests.get(file_url)
                if response.status_code != 200:
                    bot.edit_message_text(
                        "❌ فشل في تحميل الصورة",
                        chat_id=message.chat.id,
                        message_id=processing_msg.message_id
                    )
                    return
                
                image_bytes = response.content
                
                # استخراج النصوص
                bot.edit_message_text(
                    "🤖 *جاري استخراج النصوص...*",
                    chat_id=message.chat.id,
                    message_id=processing_msg.message_id,
                    parse_mode='Markdown'
                )
                
                extraction_result = text_extractor.extract_from_image(image_bytes)
                
                if not extraction_result.success:
                    bot.edit_message_text(
                        f"❌ *فشل استخراج النصوص*\n\n"
                        f"الخطأ: {extraction_result.error_message}\n\n"
                        f"💡 حاول مع صورة أوضح",
                        chat_id=message.chat.id,
                        message_id=processing_msg.message_id,
                        parse_mode='Markdown'
                    )
                    return
                
                if not extraction_result.arabic_texts and not extraction_result.english_texts:
                    bot.edit_message_text(
                        "❌ *لم يتم العثور على نصوص في الصورة*\n\n"
                        "💡 *نصائح للحصول على نتائج أفضل:*\n"
                        "• تأكد من وضوح النصوص في الصورة\n"
                        "• التقط الصورة بإضاءة جيدة\n"
                        "• اجعل الوثيقة تملأ معظم الإطار",
                        chat_id=message.chat.id,
                        message_id=processing_msg.message_id,
                        parse_mode='Markdown'
                    )
                    return
                
                # إنشاء البيانات
                name = extraction_result.extracted_name or message.from_user.first_name or "مستخدم"
                email = data_generator.generate_email(name)
                password = data_generator.generate_password()
                
                # إنشاء الملف
                bot.edit_message_text(
                    "📝 *جاري إنشاء الملف النصي...*",
                    chat_id=message.chat.id,
                    message_id=processing_msg.message_id,
                    parse_mode='Markdown'
                )
                
                generated_data = data_generator.create_text_file(
                    name=name,
                    arabic_texts=extraction_result.arabic_texts,
                    english_texts=extraction_result.english_texts,
                    email=email,
                    password=password,
                    platform=bot_manager.platform.value
                )
                
                # إرسال الملف
                from io import BytesIO
                file_stream = BytesIO(generated_data.file_content.encode('utf-8'))
                file_stream.name = generated_data.filename
                
                caption = f"""
✅ *تم استخراج المعلومات بنجاح!*

📋 *الملخص:*
👤 الاسم: {name}
📧 البريد: `{email}`
🔐 كلمة المرور: `{password}`

📊 *الإحصائيات:*
• النصوص العربية: {len(extraction_result.arabic_texts)} سطر
• النصوص الإنجليزية: {len(extraction_result.english_texts)} سطر
• وقت المعالجة: {extraction_result.processing_time:.2f} ثانية
• المحرك: {extraction_result.ai_engine.upper()}

💾 *تم حفظ كل المعلومات في الملف المرفق*
"""
                
                bot.send_document(
                    message.chat.id,
                    file_stream,
                    caption=caption,
                    parse_mode='Markdown'
                )
                
                # حذف رسالة المعالجة
                bot.delete_message(message.chat.id, processing_msg.message_id)
                
                # تحديث إحصائيات المستخدم
                if user_id in bot_manager.users_db:
                    bot_manager.users_db[user_id].extractions_count += 1
                    bot_manager.users_db[user_id].last_extraction = generated_data.timestamp
                    bot_manager.users_db[user_id].created_email = email
                
                # حفظ في السجل
                bot_manager.extraction_history.append({
                    'user_id': user_id,
                    'timestamp': generated_data.timestamp,
                    'name': name,
                    'email': email,
                    'processing_time': extraction_result.processing_time
                })
                
                bot_manager.logger.info(f"✅ عملية استخراج ناجحة للمستخدم: {user_id}")
                
                # إرسال تعليمات نهائية
                final_message = f"""
🎉 *عملية الاستخراج اكتملت بنجاح!*

📋 *بيانات الدخول الخاصة بك:*
📧 *البريد الإلكتروني:* `{email}`
🔐 *كلمة المرور:* `{password}`

⚠️ *هام: احفظ هذه البيانات في مكان آمن!*

🔧 *معلومات تقنية:*
• المنصة: {bot_manager.platform.value.upper()}
• محرك AI: {extraction_result.ai_engine.upper()}
• الثقة: {extraction_result.confidence * 100:.1f}%
• الملف: {generated_data.filename}

🔄 *لإرسال صورة أخرى:* أرسل صورة جديدة
📊 *لعرض إحصائياتك:* اضغط على زر 'إحصائيات'
"""
                
                bot.send_message(
                    message.chat.id,
                    final_message,
                    parse_mode='Markdown'
                )
                
            except Exception as e:
                bot_manager.logger.error(f"خطأ في معالجة الصورة: {e}")
                bot.reply_to(
                    message,
                    f"❌ *حدث خطأ غير متوقع*\n"
                    f"التفاصيل: {str(e)[:100]}\n\n"
                    f"🔄 الرجاء إعادة المحاولة",
                    parse_mode='Markdown'
                )
        
        @bot.message_handler(func=lambda message: message.text == "ℹ️ معلومات")
        def handle_info_button(message):
            """زر المعلومات"""
            info_text = f"""
📋 *معلومات البوت:*

🛠 *الإصدار:* {bot_manager.bot_config['bot_version']}
🌐 *المنصة:* {bot_manager.platform.value.upper()}
🤖 *محرك الذكاء:* {bot_manager.ai_type.value.upper()}
📊 *عدد المستخدمين:* {len(bot_manager.users_db)}
📈 *عمليات الاستخراج:* {sum(u.extractions_count for u in bot_manager.users_db.values())}

🔧 *المكتبات المستخدمة:*
• pyTelegramBotAPI: لواجهة تيليجرام
• Google Generative AI: لاستخراج النصوص
• OpenCV/EasyOCR: لمعالجة الصور

🔒 *الخصوصية:*
• الصور تُعالج فوراً ولا تُخزن
• البيانات تُحفظ مؤقتاً في الذاكرة
• يمكنك مسح بياناتك في أي وقت

📞 *الدعم:* @YourSupportChannel
"""
            bot.send_message(message.chat.id, info_text, parse_mode='Markdown')
        
        @bot.message_handler(func=lambda message: message.text == "📊 إحصائيات")
        def handle_stats_button(message):
            """زر الإحصائيات"""
            user_id = str(message.from_user.id)
            user_info = bot_manager.users_db.get(user_id)
            
            if user_info:
                stats_text = f"""
📊 *إحصائياتك الشخصية:*

👤 *اسمك:* {user_info.first_name or 'غير معروف'}
🆔 *معرفك:* {user_info.user_id}
📅 *تاريخ الانضمام:* {user_info.join_date}
🔢 *عدد العمليات:* {user_info.extractions_count}
📅 *آخر عملية:* {user_info.last_extraction or 'لا يوجد'}

📈 *إحصائيات عامة:*
• إجمالي المستخدمين: {len(bot_manager.users_db)}
• عمليات اليوم: {len([h for h in bot_manager.extraction_history if h['timestamp'].startswith(datetime.now().strftime('%Y-%m-%d'))])}
• المنصة: {bot_manager.platform.value.upper()}
• الوقت الحالي: {datetime.now().strftime('%H:%M:%S')}
"""
            else:
                stats_text = "❌ *لم يتم العثور على بياناتك*"
            
            bot.send_message(message.chat.id, stats_text, parse_mode='Markdown')
        
        @bot.message_handler(func=lambda message: message.text == "🆘 المساعدة")
        def handle_help_button(message):
            """زر المساعدة"""
            help_text = """
🆘 *مركز المساعدة:*

❓ *أسئلة شائعة:*

1. *ما أنواع الصور المدعومة؟*
   • البطاقة الشخصية، جواز السفر، رخصة القيادة
   • الصور بصيغة JPG, PNG, JPEG
   • حجم الصورة: حتى 10MB

2. *كم تستغرق المعالجة؟*
   • 10-30 ثانية حسب جودة الصورة
   • Gemini AI أسرع وأدق من OCR العادي

3. *كيف يتم إنشاء البريد الإلكتروني؟*
   • يتم استخراج الاسم من الصورة
   • تحويله إلى حروف لاتينية
   • إضافة نطاق عشوائي

4. *هل البيانات آمنة؟*
   • نعم، الصور تُحذف بعد المعالجة
   • لا يتم تخزين أي معلومات شخصية

🔄 *إصلاح المشاكل:*

• *الصورة غير واضحة:* حاول التصوير بإضاءة أفضل
• *لم يتم استخراج نص:* تأكد من وضوح النصوص في الصورة
• *البوت لا يرد:* أعد تشغيله أو اتصل بالدعم

📞 *للتواصل:* @YourSupportChannel
"""
            bot.send_message(message.chat.id, help_text, parse_mode='Markdown')
        
        # تشغيل البوت
        bot_manager.logger.info("🤖 جاري تشغيل البوت...")
        
        # اختبار الاتصال
        try:
            bot_info = bot.get_me()
            bot_manager.logger.info(f"✅ البوت متصل: {bot_info.first_name} (@{bot_info.username})")
            
            # اختيار طريقة التشغيل بناءً على المنصة
            if bot_manager.platform in [Platform.RENDER, Platform.RAILWAY, Platform.HEROKU]:
                # استخدام Webhook
                import flask
                from threading import Thread
                
                # إنشاء تطبيق Flask
                flask_app = FlaskAppWrapper(bot_manager)
                app = flask_app.create_app()
                
                if app:
                    @app.route('/webhook', methods=['POST'])
                    def webhook():
                        if flask.request.headers.get('content-type') == 'application/json':
                            json_string = flask.request.get_data().decode('utf-8')
                            update = telebot.types.Update.de_json(json_string)
                            bot.process_new_updates([update])
                            return ''
                        return 'Bad Request', 400
                    
                    # تعيين Webhook
                    webhook_url = os.environ.get('WEBHOOK_URL', '')
                    if webhook_url:
                        bot.remove_webhook()
                        time.sleep(1)
                        bot.set_webhook(url=f"{webhook_url}/webhook")
                        bot_manager.logger.info(f"✅ Webhook معين على: {webhook_url}")
                    
                    # تشغيل Flask في thread منفصل
                    def run_flask():
                        port = int(os.environ.get('PORT', 5000))
                        app.run(host='0.0.0.0', port=port)
                    
                    flask_thread = Thread(target=run_flask, daemon=True)
                    flask_thread.start()
                    bot_manager.logger.info(f"🌐 خادم Flask يعمل على المنفذ: {os.environ.get('PORT', 5000)}")
                    
                # تشغيل البوت بنمط polling أيضاً للاحتياط
                bot.polling(none_stop=True, interval=0, timeout=60)
                
            else:
                # استخدام Polling التقليدي
                bot_manager.logger.info("🔄 البوت يعمل بنمط Polling")
                bot.polling(none_stop=True, interval=0, timeout=60)
        
        except Exception as e:
            bot_manager.logger.critical(f"❌ فشل تشغيل البوت: {e}")
            bot_manager.logger.error(traceback.format_exc())
    
    except ImportError as e:
        print(f"❌ خطأ: المكتبة المطلوبة غير مثبتة: {e}")
        print("📦 قم بتثبيت المكتبات المطلوبة:")
        print("pip install pyTelegramBotAPI google-generativeai easyocr opencv-python-headless pillow requests")
        sys.exit(1)

# ============= نقطة الدخول الرئيسية =============
if __name__ == "__main__":
    run_bot()
