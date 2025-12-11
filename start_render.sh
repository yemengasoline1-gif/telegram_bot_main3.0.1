#!/bin/bash
echo "🚀 بدء البوت على Render.com..."
echo "📅 الوقت: $(date)"
echo "🌐 المنصة: Render"
echo "🤖 البوت: استخراج النصوص"

# تنشيط البيئة الافتراضية
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# تثبيت المتطلبات
pip install -r requirements.txt --upgrade

# تشغيل البوت
python telegram_bot_main.py
