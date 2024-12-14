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

# Méthode du gradient conjugué (version Polak-Ribiere)
def conjugate_gradient(J, grad_J, x0, y0, max_iter=1000, tol=1e-6):
    x, y = x0, y0
    u = np.array([x, y])
    grad_u = grad_J(x, y)
    
    # Initialisation de d0 et β0
    d = -grad_u
    history = [u]  # Historique des points

    for k in range(max_iter):
        # Critère d'arrêt : norme du gradient
        if np.linalg.norm(grad_u) < tol:
            break
        
        # Calcul de βk (Polak-Ribiere)
        if k == 0:
            beta = 0  # Au début, on prend β0 = 0
        else:
            beta = np.dot(grad_u, grad_u - grad_u_old) / np.linalg.norm(grad_u_old)**2

        # Calcul de la nouvelle direction dk
        d = -grad_u + beta * d
        
        # Recherche du pas optimal αk (par descente de gradient simple)
        alpha = -np.dot(grad_u, d) / np.dot(d, grad_J(u[0] + d[0], u[1] + d[1]))
        
        # Mise à jour de la solution
        u_new = u + alpha * d
        grad_u_old = grad_u
        grad_u = grad_J(u_new[0], u_new[1])

        # Ajout du point à l'historique
        history.append(u_new)
        
        # Mise à jour de u pour la prochaine itération
        u = u_new

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
    ax.set_title('Méthode du Gradient Conjugué (Polak-Ribiere)')
    ax.legend()
    
    plt.show()

# Paramètres de la méthode du gradient conjugué
x0, y0 = -1, 1  # Point initial
max_iter = 1000  # Nombre maximal d'itérations
tol = 1e-6       # Tolérance pour la convergence

# Exécution de la méthode du gradient conjugué
history_conjugate_gradient = conjugate_gradient(J, grad_J, x0, y0, max_iter, tol)

# Visualisation
plot_surface_and_history(history_conjugate_gradient)

# Affichage de l'historique des points générés
print("Historique des points générés par la méthode du Gradient Conjugué :")
print(history_conjugate_gradient)
