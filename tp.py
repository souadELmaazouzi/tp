import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Définir la fonction J(x, y)
def J(x, y):
    return (x - 1)**2 + 10 * (x**2 - y)**2
# Générer les points (x, y) pour une grille
x = np.linspace(-2, 2, 200)  # Plus de points pour une visualisation lisse
y = np.linspace(-1, 3, 200)
X, Y = np.meshgrid(x, y)     # Grille des points
Z = J(X, Y)                  # Calcul de la fonction
# --- Visualisation 3D ---
fig = plt.figure(figsize=(14, 7))
ax1 = fig.add_subplot(121, projection='3d')  # Sous-plot pour le 3D
surface = ax1.plot_surface(
    X, Y, Z, 
    cmap='viridis', 
    edgecolor='k',  # Bordures pour rendre les contours visibles
    alpha=0.9
)
# Ajouter une barre de couleurs pour montrer l'échelle des hauteurs
cbar_3d = fig.colorbar(surface, ax=ax1, shrink=0.5, aspect=10)
cbar_3d.set_label("Valeurs de J(x, y)")
# Réglages pour la vue 3D
ax1.view_init(elev=30, azim=135)  # Angle de vue pour mieux voir la surface
ax1.set_title("Visualisation 3D de la fonction J(x, y)", fontsize=12)
ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax1.set_zlabel("J(x, y)")
# --- Visualisation 2D (Lignes de niveau) ---
ax2 = fig.add_subplot(122)  # Sous-plot pour le 2D
contour = ax2.contourf(
    X, Y, Z, 
    levels=np.linspace(Z.min(), Z.max(), 50),  # 50 niveaux pour une meilleure précision
    cmap='viridis'
)
# Ajouter les contours pour les lignes de niveau
lines = ax2.contour(X, Y, Z, levels=10, colors='k', linewidths=0.8)  # Lignes noires supplémentaires
ax2.clabel(lines, inline=True, fontsize=8)  # Ajouter des étiquettes sur les lignes de contour
# Ajouter une barre de couleurs pour la 2D
cbar_2d = fig.colorbar(contour, ax=ax2)
cbar_2d.set_label("Valeurs de J(x, y)")
# Ajouter des annotations pour le titre et les axes
ax2.set_title("Lignes de niveau de J(x, y)", fontsize=12)
ax2.set_xlabel("x")
ax2.set_ylabel("y")
# Ajustement général
plt.tight_layout()
plt.show()
