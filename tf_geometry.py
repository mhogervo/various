#!/usr/bin/env python3
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def compute_metric_tf(coords, embedding, ambient_metric=None):
    """
    Given a set of coordinates x^mu, an embedding X^A(x^mu), and an ambient
    metric eta_{AB}, compute the projected metric numerically.
    If no ambient metric is specified, the standard Euclidean metric is used.

    Returns: (metric, jacobian, X^A).
    """
    
    coords = tf.Variable(coords)

    if ambient_metric is None:
        dim = coords.shape[0]
        ambient_metric = flat_metric(dim+1)

    with tf.GradientTape() as tape:
        X = embedding(coords)
        
    jac = tape.jacobian(X, coords)
    metric = tf.transpose(jac) @ ambient_metric @ jac

    return metric, jac, X


def flat_metric(p: int, q: int = 0) -> tf.Tensor:
    """
    Return a flat metric of signature q times -1 + p times +1.
    """
    
    diag = np.array(q * [-1] + p * [1], dtype='float64')
    return tf.linalg.diag(diag)


def plot_matrix(tens) -> None:
    """
    Given a matrix, return its plot.
    """
    plt.matshow(tens.numpy(), cmap=plt.get_cmap('gist_yarg'))


# Functions needed to compute the metric of the sphere S^n.
def sphere_embedding(angles) -> tf.Tensor:
    """
    Given n angles (theta_1, theta_2, ..., phi), return the embedding into R^{n+1}
    of unit norm.
    """
    n = angles.shape[0]
    if n == 1:
        phi = angles[0]
        return tf.convert_to_tensor([tf.cos(phi), tf.sin(phi)])
    elif n >= 2:
        theta, other_angles = angles[0], angles[1:]
        other_components = tf.sin(theta) * sphere_embedding(other_angles)
        return tf.concat(([tf.cos(theta)], other_components), axis=0)


def sphere_metric_tf(angles, check=False):
    """
    Given n angles (theta_1, theta_2, ..., phi), return the
    metric g_{S^n} evaluated at the point x.
    If check = True, check that the gradient is orthogonal to X itself.
    """

    metric, jac, X = compute_metric_tf(angles, sphere_embedding)
    
    if check:
        # this represents X.(grad X) - it should vanish since X.X = 1.
        err = tf.expand_dims(X, axis=0) @ jac
        print(f"This should vanish: {tf.reduce_sum(tf.square(err))}.")
        
    return metric


# Same, but for Euclidean AdS in global coordinates.
def AdS_embedding(coords) -> tf.Tensor:
    """
    Given n coordinates (rho, theta_1, theta_2, ..., phi), return the embedding X ~ R^{1,n}
    of norm X.eta.X = -1.
    Sematically, this is
        X^A = (cosh(rho), sinh(rho) m)
    where m is a unit vector in R^n, specified by n-1 angles.
    """
    n = coords.shape[0]
    if n == 1:
        rho = x[0]
        return tf.convert_to_tensor([tf.cosh(rho), tf.sinh(rho)])
    elif n >= 2:
        rho, angles = coords[0], coords[1:]
        other_components = tf.sinh(rho) * sphere_embedding(angles)
        return tf.concat(([tf.cosh(rho)], other_components), axis=0)


def AdS_metric_tf(coords, check=False):
    """
    Given n coordinates = (rho, theta_1, theta_2, ..., phi), return the
    metric g_{AdS_{n}} evaluated at that point.
    If check = True, check that the gradient is orthogonal to X itself.
    """

    dim = coords.shape[0]
    ambient_metric = flat_metric(dim, 1)

    metric, jac, X = compute_metric_tf(coords, AdS_embedding, ambient_metric)
    
    if check:
        # this represents X.eta.(grad X), where eta is the embedding space metric.
        # It should vanish since X.eta.X = -1.
        err = tf.expand_dims(X, axis=0) @ eta @ jac
        print(f"This should vanish: {tf.reduce_sum(tf.square(err))}.")
        
    return metric


# Analytically computed metrics:
def sphere_metric_analytic(angles) -> tf.Tensor:
    """
    Given angles, compute the metric analytically.
    """
    angles = list(angles.numpy())
    angles.pop()  # remove the last angle, phi - doesn't contribute to the metric.
    x = 1.
    diag = [x]
    while angles:
        theta = angles.pop(0)
        x *= tf.sin(theta)**2
        diag.append(x)
    mat = tf.linalg.diag(diag)
    return tf.cast(mat, 'float64')


def AdS_metric_analytic(coords) -> tf.Tensor:
    """
    Given coordinates (rho, ...), compute the AdS metric analytically.
    """
    coords = list(coords.numpy())
    if len(coords) < 2:
        raise ValueError(f"Input is {len(coords)} coordinates, but we need to be in dimension >= 2.")
    
    coords.pop()   # remove the last angle, phi - doesn't contribute to the metric.
    rho = coords.pop(0)
    x = tf.sinh(rho)**2
    diag = [1., x]
    while coords:
        theta = coords.pop(0)
        x *= tf.sin(theta)**2
        diag.append(x)
    mat = tf.linalg.diag(diag)
    return tf.cast(mat, 'float64')

####################


print("------------\n")
print("An example for Euclidean AdS:\n")
# convention: dim refers to the manifold AdS_{dim+1}.
dim_AdS = 4

# draw some random coordinates:
rand_coords = tf.random.uniform((dim_AdS+1,), minval=-np.pi, maxval=np.pi, dtype='float64')

g_tf = AdS_metric_tf(rand_coords)
print(f"The metric of a unit-radius AdS_{dim_AdS + 1} computed using GradientTape is \n\n{g_tf}.")

g_an = AdS_metric_analytic(rand_coords)
eps = tf.reduce_sum(tf.square(g_tf-g_an))
print(f"\nThe L^2 error between the exact and the numerical result is {eps}.")

# can use this to plot the metric:
# [plot_matrix(g_tf)]

print("\n------------\n")
####################


print("An example for the sphere:\n")
# convention: dim refers to the manifold S^dim.
dim_sphere = 3

# draw some random angles:
rand_angles = tf.random.uniform((dim_sphere,), minval=0, maxval=2*np.pi, dtype='float64')

g_tf = sphere_metric_tf(rand_angles)
print(f"The metric of the unit sphere S^{dim_sphere} computed using GradientTape is \n\n{g_tf}.")

g_an = sphere_metric_analytic(rand_angles)
eps = tf.reduce_sum(tf.square(g_tf-g_an))
print(f"\nThe L^2 error between the exact and the numerical result is {eps}.")

# can use this to plot the metric:
# [plot_matrix(g_tf)]
print("\n------------")
