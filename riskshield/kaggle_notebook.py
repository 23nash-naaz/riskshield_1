# Paste into a Kaggle notebook cell. Add the IEEE-CIS dataset via
# "+ Add Input" -> search "IEEE-CIS Fraud Detection" -> Add.
# Internet OFF is fine: lightgbm, torch, networkx are preinstalled.

!git clone -q https://github.com/YOURUSER/riskshield.git /kaggle/working/riskshield
%cd /kaggle/working/riskshield
!python run.py

# Or, if you uploaded the folder as a Kaggle Dataset instead of cloning:
# !cp -r /kaggle/input/riskshield-code/* /kaggle/working/riskshield/
