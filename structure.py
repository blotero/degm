import matplotlib.pyplot as plt
import numpy as np
import time


def get_pdf_from_r(r, mu_p = 33.9, sigma_p = 1.21, mu_n = 61, sigma_n = 1.33):   
  tp = np.exp( ( -np.log(( r / 10 ) ** (1/0.65)) / mu_p )**2 / ( 2 * sigma_p**2) ) / (np.sqrt(2 * np.pi) * r * 0.65 * sigma_p)  
  tn = np.exp( ( -np.log(( r / 10 ) ** (1/0.65)) / mu_n )**2 / ( 2 * sigma_n**2) ) / (np.sqrt(2 * np.pi) * r * 0.65 * sigma_n) 
  
  return 0.1  * tp + 0.9 * tn     



class Structure():
  def __init__(self, w, l, h, N):
    self.w = w    
    self.l = l
    self.h = h
    self.N = N

  def include_roof(self, extra):
    # Make data.
    self.extra = extra
    ind = int(self.N/2)
    self.Z[:,:ind,-1] = 2*extra*self.X[:,:ind,-1]/self.w + extra + self.h
    self.Z[:,ind:,-1] = -2*extra*self.X[:,ind:,-1]/self.w + extra + self.h
    self.populate_tensors()
    self.calculate_source_space_with_roof()    

  def build(self):
    self.set_space_points()
    self.calculate_source_space()

  def plot_external_surfaces(self, ax, alpha=0.4):  
    ax.plot_surface(self.X[:,:,-1],self.Y[:,:,-1],self.Z[:,:,-1], alpha=alpha, color='gray')
    ax.plot_surface(self.X[:,-1,:],self.Y[:,-1,:],self.Z[:,-1,:], alpha=alpha, color='gray')
    ax.plot_surface(self.X[-1,:,:],self.Y[-1,:,:],self.Z[-1,:,:], alpha=alpha, color='gray')

    ax.plot_surface(self.X[0,:,:],self.Y[0,:,:],self.Z[0,:,:], alpha=alpha, color='gray')
    ax.plot_surface(self.X[:,0,:],self.Y[:,0,:],self.Z[:,0,:], alpha=alpha, color='gray')
    ax.plot_surface(self.X[:,:,0],self.Y[:,:,0],self.Z[:,:,0], alpha=alpha, color='gray')

  def set_space_points(self):
    self.x = np.linspace(-self.w/2,self.w/2,self.N)
    self.y = np.linspace(-self.l/2,self.l/2,self.N)
    self.z = np.linspace(-self.h/2,self.h/2,self.N)
    self.X,self.Y,self.Z = np.meshgrid(self.x,self.y,self.z)
    self.populate_tensors()

  def calculate_source_space_with_roof(self):
    ind = int(self.source_X[0].size/2)
    self.source_Z[:,:ind,-1] = 2*self.extra*self.source_X[:,:ind,-1]/self.w + (self.extra + self.h)*2
    self.source_Z[:,ind:,-1] = -2*self.extra*self.source_X[:,ind:,-1]/self.w + (self.extra + self.h)*2
    self.source_space = np.array([self.source_X,self.source_Y,self.source_Z]).transpose(1,2,3,0)[:,:,0,:]    


  def calculate_source_space(self):
    scale = 1.5
    x = np.linspace(-self.w/2*scale,self.w/2*scale,self.N*2)
    y = np.linspace(-self.l/2*scale,self.l/2*scale,self.N*2)
    z = 1.5*self.h
    self.source_X,self.source_Y,self.source_Z = np.meshgrid(x,y,z)
    #self.source_X = np.copy(self.X)
    #self.source_Y = np.copy(self.Y)
    #self.source_Z = np.copy(self.Z) + self.h/2    
    self.source_space = np.array([self.source_X,self.source_Y,self.source_Z]).transpose(1,2,3,0)[:,:,0,:]    

  def set_geometry(self, width, lenght, height):
    self.w = width
    self.l = lenght
    self.h = height
    self.populate_tensors()

  def include_corner_rods(self, rod_height):
    """
    Include rods in the structures corner
    """
    self.Z[0,0,-1] += rod_height
    self.Z[0,-1,-1] += rod_height
    self.Z[-1,0,-1] += rod_height
    self.Z[-1,-1,-1] += rod_height
    self.populate_tensors()

  def add_rod(self, x, y, height):
    self.Z[x,y,-1] += height
    self.populate_tensors() 
    

  def populate_tensors(self):
    self.tensor = np.array([self.X,self.Y,self.Z]).transpose(1,2,3,0)
    self.sup = self.tensor[:,:,self.N-1,:]
    
  def plot_structure(self):
    fig = plt.figure(figsize=(20,10))    
    ax = fig.add_subplot(projection='3d')
    ax.scatter(self.X[:,:,-1],self.Y[:,:,-1],self.Z[:,:,-1],c='red', label='Puntos de interés de superficie superior')
    #ax.scatter(self.X[:,:,:-1],self.Y[:,:,:-1],self.Z[:,:,:-1],c='green', label='Puntos discretos de volumen interior')
    ax.scatter(self.source_X,self.source_Y,self.source_Z,c='blue', label='Puntos fuente por esfera rodante')
    self.plot_external_surfaces(ax)
    ax.legend()
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_zlabel('$z$')
    ax.set_title("Estructura cuboide a analizar")

  def find_nearest(self,p):
    lowest = 3*self.h
    index = []    
    for a in range(self.sup.shape[0]):
      for b in range(self.sup.shape[1]):        
        r = np.linalg.norm(p-self.sup[a,b])         
        if r<lowest:
          index = []          
          lowest = np.copy(r)
          index.append([a,b])
        elif r==lowest:
          if r<=lowest:    
            lowest = np.copy(r)
            index.append([a,b])
        
    return lowest, index

  def calculate_pdf(self):
    print("Calculando...")
    init_time = time.time()
    pdf = np.zeros((self.N,self.N))
    self.pdf_per_point = np.zeros((self.N,self.N))
    for i in range(self.source_space.shape[0]):
      for j in range(self.source_space.shape[1]):        
        r, index = self.find_nearest(self.source_space[i,j])
        if r>0:          
          for a,b in index:
            pdf[a,b] += get_pdf_from_r(r)
    self.pdf_per_point = pdf/pdf.sum()
    end_time = time.time()
    interval = ( end_time -init_time )
    niter = self.source_space.shape[0]*self.source_space.shape[1]*self.N**2
    print(f"Calculado con éxito tras {niter} iteraciones en {interval:.3f} segundos")

  def plot_from_top(self):
    fig = plt.figure(figsize=(20,10))
    sca = plt.scatter(self.X[:,:,-1], self.Y[:,:,-1], c=self.pdf_per_point)
    plt.grid()
    cbar = fig.colorbar(sca, shrink=0.5, aspect=5)
    cbar.set_label("Probabilidad de impacto")
    plt.show()


  def plot_pdf(self):
    fig = plt.figure(figsize=(20,10))      
    ax = fig.add_subplot(projection='3d')
    ax.autoscale(True)
    
    sca = ax.scatter(self.X[:,:,-1],self.Y[:,:,-1],self.Z[:,:,-1], c=self.pdf_per_point, label='Puntos de interés de superficie superior', marker='o', s=40)
    
    #ax.scatter(self.X[0,0,-1],self.Y[0,0,-1],self.Z[0,0,-1], color='red', label=f'P={self.pdf_per_point[0,0]:.2f}', marker='X', s=100)
    #ax.scatter(self.X[0,-1,-1],self.Y[0,-1,-1],self.Z[0,-1,-1], color='red', marker='X', s=100)     
    #ax.scatter(self.X[-1,0,-1],self.Y[-1,0,-1],self.Z[-1,0,-1], color='red', marker='X', s=100)     
    #ax.scatter(self.X[-1,-1,-1],self.Y[-1,-1,-1],self.Z[-1,-1,-1], color='red', marker='X', s=100)     
    
    ax.legend()
    self.plot_external_surfaces(ax)
    cbar = fig.colorbar(sca, shrink=0.5, aspect=5)
    cbar.set_label("Probabilidad de impacto")
