"""
This python package contains codes for solving the 2D heat equation on a square using
the Finite difference methods.
"""

#Importing the dependecies
import numpy as np                  #For working with matrices
import matplotlib.pyplot as plt     #For visualiztion
from pydantic import BaseModel , field_validator , FieldValidationInfo     #For handeling config data efficiently

#___________________________________________________________________________
#Configs: Handeling the boundary conditions.
#___________________________________________________________________________

class ThermalBoundaryConditions(BaseModel):
    """Defines the thermal boundary conditions for a rectangular domain.

    Parameters
    ----------
    length : float
        Length of the domain (x-direction) in meters.
    width : float
        Width of the domain (y-direction) in meters.
    dx : float
        Spatial discretization step in x-direction (must be > 0).
    dy : float
        Spatial discretization step in y-direction (must be > 0).
    T_top : np.ndarray
        Temperature values at top boundary (y=width) in kelvins.
    T_bottom : np.ndarray
        Temperature values at bottom boundary (y=0) in kelvins.
    T_left : np.ndarray
        Temperature values at left boundary (x=0) in kelvins.
    T_right : np.ndarray
        Temperature values at right boundary (x=length) in kelvins.

    Raises
    ------
    ValueError
        If boundary condition arrays have incorrect lengths.
        If dx or dy are not positive.
    """

    #Mesh size
    dx : float  #[m]
    dy : float  #[m]

    #Boundary condition on the top border
    T_top   : list  #[k]
    T_right : list  #[k]
    T_left  : list  #[k]
    T_bottom: list  #[k]

    #The legnth and width
    length : float = 1  #[m]
    width  : float = 1  #[m]


    @field_validator('dx', 'dy')
    @classmethod
    def validate_discretization_steps(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"Discretization step must be positive, got {v}")
        return v

    @field_validator('T_top', 'T_bottom')
    @classmethod
    def validate_top_bottom_lengths(cls, v: np.ndarray, info: FieldValidationInfo) -> np.ndarray:
        data = info.data
        if 'length' in data and 'dx' in data:
            expected_length = int(data['length'] / data['dx']) - 1
            if len(v) != expected_length:
                raise ValueError(
                    f"Expected length {expected_length} for top/bottom boundaries, got {len(v)}"
                )
        return v

    @field_validator('T_left', 'T_right')
    @classmethod
    def validate_left_right_lengths(cls, v: np.ndarray, info: FieldValidationInfo) -> np.ndarray:
        data = info.data
        if 'width' in data and 'dy' in data:
            expected_length = int(data['width'] / data['dy']) - 1
            if len(v) != expected_length:
                raise ValueError(
                    f"Expected length {expected_length} for left/right boundaries, got {len(v)}"
                )
        return v


#___________________________________________________________________________
#Solver: Solving the heat equation on the specified domain with the intial conditions provided.
#___________________________________________________________________________

def Solver(bond_cond:ThermalBoundaryConditions) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve the 2D steady-state heat conduction problem using finite difference method.

    Parameters
    ----------
    bond_cond : ThermalBoundaryConditions
        An instance of ThermalBoundaryConditions containing:
        - Domain dimensions (length, width)
        - Discretization steps (dx, dy)
        - Temperature boundary conditions (T_top, T_bottom, T_left, T_right) all in kelvins.

    Returns
    -------
    tuple
        A tuple containing three elements:
        - X : numpy.ndarray
            Solution vector of temperatures at interior points (flattened)
        - A : numpy.ndarray
            Coefficient matrix of the linear system
        - Y : numpy.ndarray
            Constant vector representing boundary conditions (flattened and scaled)

    Notes
    -----
    The solver implements the following approach:
    1. Sets up a coefficient matrix representing the finite difference equations
    2. Constructs a constant vector from boundary conditions
    3. Solves the linear system AX = Y using numpy.linalg.solve

    The finite difference scheme uses:
    - Central differencing for interior points
    - Direct application of boundary conditions
    - 5-point stencil (current point + 4 neighbors)

    Example
    -------
    >>> conditions = ThermalBoundaryConditions(
    ...     length=1,
    ...     width=1,
    ...     dx=0.25,
    ...     dy=0.25,
    ...     T_top=[500, 500, 500],
    ...     T_bottom=[356.9, 229.05, 365.99],
    ...     T_left=[500, 500, 500],
    ...     T_right=[500, 500, 500]
    ... )
    >>> X, A, Y = Solver(conditions)
    """
    

    #Determining the number of unkonws 
    x_vars = len(bond_cond.T_top)
    y_vars = len(bond_cond.T_right)
    num_vars = x_vars * y_vars
    
    #==Coefficent matrix==
    A = np.eye(num_vars)

    for var in range(num_vars):
        x ,y = __var_ind_to_xy(var, x_vars)
        
        if x + 1 < x_vars:
            A[var, __xy_to_m(x+1,y,x_vars)] = -0.25 #➡️ Right node
        if x - 1 >= 0:
            A[var, __xy_to_m(x-1,y,x_vars)] = -0.25 #⬅️ Left node

        if y + 1 < y_vars:
            A[var, __xy_to_m(x,y+1,x_vars)] = -0.25 #⬆️ Top node
        if y - 1 >= 0:
            A[var, __xy_to_m(x,y-1,x_vars)] = -0.25 #⬇️ Bottom node
        

    #==Constant matrix==
    Y = np.zeros((y_vars, x_vars))

    #edges
    Y[0,:] += bond_cond.T_bottom
    Y[-1,:] += bond_cond.T_top
    Y[:,0] += bond_cond.T_left
    Y[:,-1] += bond_cond.T_right

    #Reshaping the Y
    Y = Y.reshape((num_vars, 1)) / 4

    #==Solving for the X:Method1==
    X1 = np.linalg.solve(A, Y)

    #==Solving for the Y:Method2==
    #⚠️This method is less optimized!!
    # A_inv = np.linalg.inv(A)
    # X2 = np.matmul(A_inv, Y) 

    return X1, A, Y

#Will convert the x and y index of the node from the square domain to the m which is index of node in solution vector.
def __xy_to_m(x:int, y:int, x_vars:int) -> tuple[int, int]:
    
    return x + y * x_vars

#Will convert the index of the node in the solution vector to the x and y index of the node in the square domain. 
def __var_ind_to_xy(m:int, x_vars:int) -> tuple[int,int]:

    x = m % x_vars
    y = m // x_vars

    return x, y

#___________________________________________________________________________
#Visualizer: Displaying the temperature distribution 
#___________________________________________________________________________

def Visualizer(temperatures:np.ndarray, bond_cond:ThermalBoundaryConditions):
    """Visualize the temperature distribution as a colored contour plot.
    
    This function takes the solved temperature array and boundary conditions,
    reconstructs the complete temperature matrix including boundary values,
    and generates a pseudocolor plot of the temperature distribution.
    
    Parameters
    ----------
    temperatures : np.ndarray
        1D array of temperature values from the solver (interior points only)
    bond_cond : ThermalBoundaryConditions
        Object containing the boundary conditions with attributes:
        - T_top: Temperature values at top boundary
        - T_bottom: Temperature values at bottom boundary
        - T_left: Temperature values at left boundary
        - T_right: Temperature values at right boundary
        
    Returns
    -------
    None
        Displays a matplotlib figure with the temperature contour plot.
        
    Notes
    -----
    The function performs the following steps:
    1. Reshapes the 1D temperature array into a 2D matrix
    2. Adds boundary conditions to all sides of the matrix
    3. Calculates corner temperatures as averages of adjacent boundaries
    4. Creates a pseudocolor plot with a jet colormap
    5. Adds a colorbar and title to the plot
        
    Examples
    --------
    >>> conditions = ThermalBoundaryConditions(
    ...     length=1,
    ...     width=1,
    ...     dx=0.25,
    ...     dy=0.25,
    ...     T_top=[500, 500, 500], 
    ...     T_bottom=[356.99, 339.05, 356.99],
    ...     T_left=[500, 500, 500],  
    ...     T_right=[500, 500, 500]
    ... )
    >>> X, A, Y = Solver(conditions)
    >>> Visualizer(X, bond_cond=conditions)
    """
    
    #==Restoring the temperature matrix==
    x_vars = np.array(bond_cond.T_top).size
    y_vars = np.array(bond_cond.T_right).size
    num_vars = x_vars * y_vars

    T = temperatures.reshape((y_vars, x_vars))
    T = np.flipud(T)

    #Appending the right and the left boundry conditions to the T 
    right = np.array(bond_cond.T_right).reshape((np.array(bond_cond.T_right).size , 1))
    left = np.array(bond_cond.T_left).reshape((np.array(bond_cond.T_left).size , 1))
    T = np.hstack([left, T, right])

    #Appending the top and bottom boundary conditions to T
    bottom = np.array(bond_cond.T_bottom)
    top = np.array(bond_cond.T_top)

    #Generating average values for the corners and adding them to the top and bottom
    top_right = (bond_cond.T_top[-1] + bond_cond.T_right[-1])/2
    top_left = (bond_cond.T_top[0] + bond_cond.T_left[-1])/2
    bottom_right = (bond_cond.T_bottom[-1] + bond_cond.T_right[0])/2
    bottom_left = (bond_cond.T_bottom[0] + bond_cond.T_left[0])/2

    
    top = np.insert(top, len(top), top_right)
    top = np.insert(top, 0, top_left)
    bottom = np.insert(bottom, len(bottom), bottom_right)
    bottom = np.insert(bottom, 0, bottom_left)

    T = np.vstack([top , T, bottom])

    #==test==
    T = np.flipud(T)
    X, Y = np.meshgrid(np.arange(0, bond_cond.length + bond_cond.dx, bond_cond.dx), np.arange(0, bond_cond.width + bond_cond.dy, bond_cond.dy))


    #==Plotting the temperature distribution as a contuour==
    fig , axis = plt.subplots()
    pcm = axis.pcolormesh(X, Y, T, cmap=plt.cm.rainbow, vmin=np.min(T) * 0.8, vmax=np.max(T) * 1.2, shading='gouraud')
    plt.colorbar(pcm, ax=axis)
    axis.set_aspect(bond_cond.width/bond_cond.length)
    plt.title("Temperature contour [k]")

