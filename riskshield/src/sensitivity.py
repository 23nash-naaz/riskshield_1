import numpy as np
from evaluate import rupees
from economics import COST

def tornado_chart(y, p, amount, base_c=COST):
    """
    Varies each constant in economics.py by +/- 50% and calculates the
    resulting swing in total cost, assuming threshold and actions are optimal
    under the base cost (or re-optimized under the new cost? - typically
    tornado varies the parameter and re-evaluates. Here we'll re-decide).
    """
    n = len(y)
    
    # Base cost
    base_rupees = rupees(y, p, amount, c=base_c)
    base_model_cost = base_rupees["model"]
    
    variations = {}
    
    for key, val in base_c.items():
        # -50%
        c_low = base_c.copy()
        c_low[key] = val * 0.5
        cost_low = rupees(y, p, amount, c=c_low)["model"]
        
        # +50%
        c_high = base_c.copy()
        c_high[key] = val * 1.5
        
        # margin shouldn't exceed 1.0
        if key == "margin":
            c_high[key] = min(val * 1.5, 0.95)
            
        # stepup_stops shouldn't exceed 1.0
        if key == "stepup_stops":
            c_high[key] = min(val * 1.5, 1.0)
            
        cost_high = rupees(y, p, amount, c=c_high)["model"]
        
        swing = abs(cost_high - cost_low)
        variations[key] = {
            "low": float(cost_low),
            "high": float(cost_high),
            "swing": float(swing)
        }
        
    # Sort by swing descending
    sorted_vars = sorted(variations.items(), key=lambda x: x[1]["swing"], reverse=True)
    
    return {
        "base_cost": float(base_model_cost),
        "variables": [{"name": k, **v} for k, v in sorted_vars]
    }
