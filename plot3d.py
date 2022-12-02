import matplotlib.pyplot as plt
from numpy import meshgrid, vectorize, linspace


def prep_3d_plot(f, x=(0, 1), y=(0, 1), N=30):
    """
    Given a function f(x,y) and ranges x[0] < x < x[1], y[0] < y < y[1],
    with N = (Nx, Ny) points in both directions,
    return the tuple (x,y,z) with z = f(x,y).
    If N is a single integer, we set Nx = Ny = N.
    """
    if isinstance(N,int):
        N = (N, N)

    xr = linspace(x[0], x[1], N[0])
    yr = linspace(y[0], y[1], N[1])
    x_mat, y_mat = meshgrid(xr, yr)
    f_vec = vectorize(f)
    z_mat = f_vec(x_mat, y_mat)
    return x_mat, y_mat, z_mat


def make_3d_plot(f, x=(0, 1), y=(0, 1), N=25, alpha=0.5, angles=(25, 25)):
    """
    Given a function f(x,y) and ranges x[0] < x < x[1], y[0] < y < y[1],
    with N points in both directions, make a plot.
    N can be a single integer or a tuple (Nx, Ny).
    """
    x_mat, y_mat, z_mat = prep_3d_plot(f, x=x, y=y, N=N)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot_surface(x_mat, y_mat, z_mat, alpha=alpha)
    ax.view_init(angles[0], angles[1])
    plt.show()


# example code:
def my_func(x, y):
    return pow(x+1, 2) + 0.01*pow(y-1/2, 7)


X, Y, Z = prep_3d_plot(my_func, x=(-1, 1), y=(-2, 2), N=40)
print("Dimensions of the data points: {}, {}, {}.".format(X.shape, Y.shape, Z.shape))
make_3d_plot(my_func, x=(-1, 1), y=(-2, 2))
