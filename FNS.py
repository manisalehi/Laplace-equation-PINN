"""
This code will numerically solve the 2D steady state heat equation using Fast Fourier Transform.
"""

#Importing the dependecies
import numpy as np                  #For working with matrices
import matplotlib.pyplot as plt     #For visualiztion
from pydantic import BaseModel , field_validator , FieldValidationInfo     #For handeling config data efficiently
from math import pi

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
    T_top : np.ndarray
        Temperature values at top boundary (y=width) in kelvins.
    T_bottom : np.ndarray
        Temperature values at bottom boundary (y=0) in kelvins.
    T_left : np.ndarray
        Temperature values at left boundary (x=0) in kelvins.
    T_right : np.ndarray
        Temperature values at right boundary (x=length) in kelvins.

    """

    #Boundary condition on the top border
    T_top   : list  #[k]
    T_right : list  #[k]
    T_left  : list  #[k]
    T_bottom: list  #[k]

    #The legnth and width
    length : float = 1  #[m]
    width  : float = 1  #[m]


#___________________________________________________________________________
#Solver:  Solving the heat equation on the specified domain with the intial conditions provided.
#___________________________________________________________________________

def Solver(bond_cond:ThermalBoundaryConditions) -> np.ndarray:

    #==Finding the Fourier coefficents==
    x_vars = len(bond_cond.T_top)
    y_vars = len(bond_cond.T_right)

    #Finding the Fourier coefficents 
    f_top = np.fft.fft(bond_cond.T_top, norm="forward")
    f_bottom = np.fft.fft(bond_cond.T_bottom, norm="forward")
    f_left = np.fft.fft(bond_cond.T_left, norm="forward")
    f_right = np.fft.fft(bond_cond.T_right, norm="forward")

    #Determining the magnitude and angle
    f_top_mag = np.abs(f_top) 
    f_bottom_mag = np.abs(f_bottom)   
    f_right_mag = np.abs(f_right)   
    f_left_mag = np.abs(f_left)   

    f_top_ang = np.angle(f_top)         #In radians
    f_bottom_ang = np.angle(f_bottom)   #In radians
    f_right_ang = np.angle(f_right)     #In radians
    f_left_ang = np.angle(f_left)       #In radians

    #Frequencies [Hz]
    freq_top = np.arange(0,x_vars,1) * 2*pi/x_vars
    freq_bottom = np.arange(0,x_vars,1) * 2*pi/x_vars
    freq_right = np.arange(0,y_vars,1) * 2*pi/y_vars
    freq_left = np.arange(0,y_vars,1) * 2*pi/y_vars    

    
def __temp_dist_func(freq:np.ndarray, mag:np.ndarray, angle:np.ndarray, s:float):
    "Calculating the temperature of a point a function of time"
    
