from matplotlib import projections
import numpy as np 
import matplotlib.pyplot as plt 
import scipy.linalg as la 


def objective(x):
    return (x[0]**2+x[1]-11)**2 + (x[0] + x[1]**2 -7)**2 


def get_unif_random(x,r,p):
    x = np.array(x, dtype=float)
    d = len(x)
    shift = np.random.uniform(low = -r , high=r, size=d)
    rnd = np.random.rand(d)
    x_new = x.copy()
    x_new[rnd<=p] = x_new[rnd<=p]+shift[rnd<=p]
    return x_new


def hill_climbing(fun, x0, r=0.001, p=1, n=1, max_iter=100000, tol=1e-10):
    x0 = np.array(x0)
    best_x = x0 
    best_f = fun(x0)
    d = len(x0)
    xs = np.zeros((1+max_iter,d))
    xs[0, :]=x0 

    n_iter = 0
    while n_iter < max_iter:
        n_iter+=1 
        x_new = np.array([get_unif_random(best_x, r,p) for _ in range(n)])
        f_val = np.apply_along_axis(fun,1,x_new)
        max_ind = np.argmax(f_val)

        if f_val[max_ind]> best_f :
            best_x = x_new[max_ind]
            best_f = f_val[max_ind]
            xs[n_iter,:] = best_x

            if np.abs(f_val[max_ind] -best_f)<=tol :
                break 

            xs[n_iter,:] = best_x

    return best_x, best_f, xs[:n_iter+1]









def main():
    r_min, r_max = -5.0, 5.0 
    X,Y = np.meshgrid(np.linspace(r_min, r_max,100),np.linspace(r_min, r_max,100))
    Z = np.c_[X.ravel(), Y.ravel()]
    results = np.apply_along_axis(objective, 1, Z)
    results = results.reshape(X.shape)
    fig = plt.figure()
    axis = fig.gca(projection = '3d')
    axis.plot_surface(X,Y, results, cmap='jet')
    #plt.contour(X,Y, results, [10,50,100,200,500], cmap='jet')
    #plt.colorbar()
   

    f_max = lambda x: -(objective(x))
    best_x, best_f, xs = hill_climbing(f_max, [0,0],r=1,n=5, max_iter=1000000)
    plt.contour(X,Y,results, [10,50,100,200,500], cmap='jet')
    print(xs[:,0], xs[:,1])
    plt.plot(xs[:,0], xs[:,1], 'r:')
    plt.show()
    
    

if __name__ == "__main__":
    main()


