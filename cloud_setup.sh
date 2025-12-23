2#!/bin/bash
# Cloud Student Backtest Setup Script

echo "🚀 Setting up Cloud Environment for Student Backtest"

# Update system
sudo apt-get update
sudo apt-get install -y python3 python3-pip git htop

# Clone repository (adjust URL)
git clone https://github.com/your-repo/crypto_portfolio_moments.git
cd crypto_portfolio_moments

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install requirements
pip install -r requirements.txt

# Install additional cloud packages
pip install psutil  # For monitoring

echo "✅ Environment ready!"

# Quick test command
echo "🔬 For QUICK TEST (recommended first):"
echo "python run_student_only.py 1D --quick"
echo ""
echo "🏁 For FULL RUN:"
echo "python run_student_only.py 1D"

# Monitor script
echo '#!/bin/bash
while true; do
    echo "=== $(date) ==="
    echo "CPU Usage:"
    top -bn1 | head -5
    echo ""
    echo "Memory Usage:"
    free -h
    echo ""
    echo "Python processes:"
    ps aux | grep python | head -5
    echo "================================"
    sleep 30
done' > monitor.sh
chmod +x monitor.sh

echo ""
echo "📊 To monitor performance: ./monitor.sh"
echo "💰 Remember to STOP instance after completion!"
