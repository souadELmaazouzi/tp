import numpy as np
import matplotlib.pyplot as plt

# Définition de la fonction J(x, y)
def J(x, y):
    return (x - 1)**2 + 10 * (x**2 - y)**2

# Définition du gradient de la fonction J(x, y)
def grad_J(x, y):
    grad_x = 2 * (x - 1) + 40 * x * (x**2 - y)
    grad_y = -20 * (x**2 - y)
    return np.array([grad_x, grad_y])

# Implémentation de la méthode du gradient à pas constant
def gradient_descent(J, grad_J, x0, y0, alpha, max_iter=1000, tol=1e-6, max_grad_norm=10):
    x, y = x0, y0
    history = [(x, y)]  # Historique des points
    
    for i in range(max_iter):
        grad = grad_J(x, y)
        
        # Contrôle de la norme du gradient
        if np.linalg.norm(grad) > max_grad_norm:
            print(f"Norme du gradient trop grande à l'itération {i}.")
            break
        
        x_new, y_new = np.array([x, y]) - alpha * grad
        
        history.append((x_new, y_new))
        
        # Arrêt si la différence entre les itérations est suffisamment petite
        if np.linalg.norm([x_new - x, y_new - y]) < tol:
            break
        
        x, y = x_new, y_new
    
    return np.array(history)

# Fonction pour afficher la surface de J(x, y) et l'historique des points
def plot_surface_and_history(history):
    # Création de la grille de points pour afficher la fonction J(x, y)
    x_vals = np.linspace(-2, 2, 400)
    y_vals = np.linspace(-2, 2, 400)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = J(X, Y)
    
    # Tracé de la surface
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.6)
    
    # Tracé des points de l'historique
    history = np.array(history)
    ax.plot(history[:, 0], history[:, 1], J(history[:, 0], history[:, 1]), color='r', marker='o', markersize=4, label="Points générés")
    
    # Paramètres du graphique
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('J(x, y)')
    ax.set_title('Méthode du gradient à pas constant')
    ax.legend()
    
    plt.show()

# Paramètres de la méthode de gradient
x0, y0 = -1, 1  # Point initial
alpha = 0.01     # Pas constant
max_iter = 1000  # Nombre maximal d'itérations
tol = 1e-6       # Tolérance pour la convergence

# Exécution de la méthode du gradient
history = gradient_descent(J, grad_J, x0, y0, alpha, max_iter, tol)

# Visualisation
plot_surface_and_history(history)

# Affichage de l'historique des points générés
print("Historique des points générés par la méthode de gradient:")
print(history)
