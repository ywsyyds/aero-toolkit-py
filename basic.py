import numpy as np

def thin_airfoil_cl(alpha_deg, alpha0_deg=0):
    """薄翼理论: Cl = 2π(α - α0), 弧度制"""
    alpha = np.radians(alpha_deg)
    alpha0 = np.radians(alpha0_deg)
    return 2 * np.pi * (alpha - alpha0)

alphas = np.linspace(-5, 15, 21)
cls = thin_airfoil_cl(alphas)
for a, cl in zip(alphas, cls):
    print(f"α={a:5.1f}°  Cl={cl:6.3f}")