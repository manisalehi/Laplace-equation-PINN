#Importing the dependecies
import numpy as np                  #For working with matrices
from FDM import Solver, ThermalBoundaryConditions   #For Dataset
import tensorflow as tf                             #For Making the nn
from tensorflow import keras                        #For Making the nn
from typing import Callable
from math import ceil

#___________________________________________________________________________
#Data Generation:Using the FDM to generate datasets
#___________________________________________________________________________

def GetDataSet(num_data:int, 
               T_min:float, 
               T_max:float, 
               length:float, 
               width:float, 
               dx:float, 
               dy:float) -> tuple[tuple[list,list,list,list, list, list] , tuple[list]]:
    """
    Generate a dataset of random thermal boundary conditions and corresponding temperature distributions.
    
    This function creates multiple sets of random boundary conditions, solves the 2D steady-state
    heat equation for each set using Finite Difference Method (FDM), and returns both the boundary
    conditions and the computed temperature fields with their coordinates.

    Parameters
    ----------
    num_data : int
        Number of simulations/datapoints to generate.
    T_min : float
        Minimum temperature value for random boundary conditions (in Kelvin).
    T_max : float
        Maximum temperature value for random boundary conditions (in Kelvin).
    length : float
        Length of the rectangular domain (x-direction).
    width : float
        Width of the rectangular domain (y-direction).
    dx : float
        Spatial discretization step in x-direction.
    dy : float
        Spatial discretization step in y-direction.

    Returns
    -------
    tuple
        (boundary_conditions, solution_data) where:
        - boundary_conditions: Tuple of (top, bottom, right, left, mesh_sizes, coordinates)
          * Each contains numpy arrays of shape (num_data, n_points_per_edge)
          * mesh_sizes: Array of [dx, dy] repeated num_data times
          * coordinates: Array of shape (n_internal_points, 2) with (x,y) positions
        - solution_data: Tuple of (temperature_fields)
          * temperature_fields: Array of shape (num_data, n_internal_points)
          
    Notes
    -----
    - The function uses ThermalBoundaryConditions and Solver for FDM calculations
    - Boundary temperatures are generated with uniform random distribution in [T_min, T_max]
    - Internal points are automatically calculated based on dx and dy spacing

    Examples
    --------
    >>> # Generate 4 simulations with temperature range 2772-3772K
    >>> num_sim = 4
    >>> T_min = 2500 + 273 - 1  # 2772K
    >>> T_max = 3500 + 273 - 1  # 3772K 
    >>> length, width = 1, 5
    >>> dx, dy = 0.1, 0.25
    >>> (top, bottom, right, left, mesh_size, coords), (temps) = GetDataSet(
    ...     num_sim, T_min, T_max, length, width, dx, dy)
    >>> 
    >>> # Access first simulation results:
    >>> print(f"First simulation top BC: {top[0]}")
    >>> print(f"First simulation temperatures: {temps[0]}")
    >>> print(f"Coordinates shape: {coords.shape}")

    See Also
    --------
    ThermalBoundaryConditions : Class handling boundary condition setup
    Solver : FDM solver for the heat equation
    """

    #==Generating random boundary conditions==
    num_data_points_x = int(length/dx -1)                           #Number of horizontal boundary conds points
    num_data_points_y = int(length/dy -1)                           #Number of vertical boundary conds points
    num_data_points = 2 * (num_data_points_x + num_data_points_y)   #Number of total boundary conds points
    
    features = np.random.randint(low = T_min, high=T_max, size=(num_data, num_data_points)).astype('float32') 
    features += np.random.rand(num_data, num_data_points)           #Adding the decimals 
    features = features.astype('float32')                           #Reducing the size of the data set

    #📦Organizing and seperating the top, bottom, right and left boundary conditions from one another
    features_top = features[:,0 : num_data_points_x]
    features_bottom = features[: , num_data_points_x : 2 * num_data_points_x]
    features_right = features[:,2 * num_data_points_x : - num_data_points_y]
    features_left = features[:,-num_data_points_y:]

    #==Solving for the temperature distribution==
    #Generating a tensor to hold the sim resualts
    #(method1)
    targets = np.empty(shape=(num_data,num_data_points_x*num_data_points_y)) 

    #(method2):⚠️Is less optimized due to inital valuation of memory
    #targets = np.zeros(shape=(num_data,num_data_points_x*num_data_points_y))

    #🏃🏻‍♂️Running FDM simulations
    for indx , (top,bottom,right,left) in enumerate(zip(features_top, features_bottom, features_right, features_left)):
        #setting up the problem's boundary conditons
        try:
            conditions = ThermalBoundaryConditions(
                length=length,
                width=width,
                dx=dx,
                dy=dy,
                T_top=top, 
                T_bottom=bottom,
                T_left=left,  
                T_right=right
            )
        except ValueError as e:
            print(f"Validation error: {e}")

        #Simulating for the specific boundary condition
        res , _, _ = Solver(bond_cond=conditions)

        #Saving the sim resualts
        targets[indx] = res.reshape((len(res), ))

    #Reshaping the targets -> vectorize
    targets = targets.reshape((targets.size))

    #==Finding the coordinates of the nodes==
    coordiantes = np.array([__var_ind_to_xy(m, num_data_points_x) for m in range(len(res))], dtype="float64")
    coordiantes *= np.array([dx, dy])
    coordiantes += np.array([dx, dy]) 

    #Vectorizing the coordinates
    coordiantes = np.repeat(coordiantes.reshape([1,coordiantes.shape[0],2]),num_data, axis=0)
    coordiantes = coordiantes.reshape([int(coordiantes.size/2), 2])

    #==Making mesh size array==
    mesh_size = np.array([[dx, dy]])
    mesh_size = np.repeat(mesh_size, repeats=len(targets), axis=0)

    #==Vectorizing the features==
    features_top = np.repeat(features_top ,repeats=len(res), axis=0)
    features_bottom = np.repeat(features_bottom ,repeats=len(res), axis=0)
    features_right = np.repeat(features_right ,repeats=len(res), axis=0)
    features_left = np.repeat(features_left ,repeats=len(res), axis=0)

    return (features_top, features_bottom, features_right, features_left, mesh_size, coordiantes), (targets)


#Will convert the index of the node in the solution vector to the x and y index of the node in the square domain. 
def __var_ind_to_xy(m:int, x_vars:int) -> tuple[int,int]:

    x = m % x_vars
    y = m // x_vars

    return x, y


#___________________________________________________________________________
#Building the model
#___________________________________________________________________________

def BuildFunc(num_hidden_layers:int,
                num_units:int,
                input_shapes:list,
                activation:str|Callable[[tf.Tensor], tf.Tensor]='tanh') -> keras.Model:
    """Builds a feedforward neural network with multiple inputs.
    
    Args:
        num_hidden_layers: Number of hidden layers
        num_units: Neurons per hidden layer
        input_shapes: List of input feature dimensions (e.g., [n_features1, n_features2])
        activation: Activation function (string or callable)
        
    Returns:
        Configured Keras Model
    """
    
    #Using a seed for consistency
    keras.utils.set_random_seed(42)

    #==Model architecture==
    
    #🚪Input layers
    inputs = [keras.layers.Input(shape=(shape,)) for shape in input_shapes]
    
    #Concatinating the input layers
    x = keras.layers.concatenate(inputs)

    #🔮Hidden layers
    for _ in range(num_hidden_layers):
        x = keras.layers.Dense(num_units, activation=activation)(x)

    #Usage of periodic activation function can boost the training process -> more sample efficient.
    #📃https://arxiv.org/abs/2212.08965
    #⚠️In this project it caused sginificant netowrk unstability problems
    # sin_activation = lambda x: tf.sin(x)
    # for _ in range(num_hidden_layers):
    #     x = keras.layers.Dense(num_units, 
    #                            activation=sin_activation,
    #                         #    kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.2), 
    #                         #    bias_initializer=keras.initializers.Constant(value=2)
    #             )(x)

    #🎁Output layer
    output = keras.layers.Dense(units=1)(x)

    #This code was used for testing of the model's ability to predict large value
    #⚠️I do not recommend to use this code because it may cause unstability in the network
    # output = keras.layers.Lambda(lambda X: -2 *X)(x)
    # output = keras.layers.Dense(units=1)(output)

    #Setting up the model
    model = keras.Model(inputs, output)

    return model


#___________________________________________________________________________
# Normaliztion/Standardization of the data
#___________________________________________________________________________

def Standardizer(training_x: list[np.ndarray]) -> tuple[list[np.ndarray], dict[str, dict[str, float]]]:
    """
    Standardizer boundary conditions and coordinates using z-score standardization.
    
    For each input feature, computes:
    - Mean (μ) and standard deviation (σ) statistics
    - Applies transformation: (x - μ) / σ
    
    Parameters
    ----------
    training_x : List[np.ndarray]
        Input data containing in order:
        [0] Top boundary conditions (1D array)
        [1] Bottom boundary conditions (1D array)
        [2] Right boundary conditions (1D array) 
        [3] Left boundary conditions (1D array)
        [4] Coordinate array of shape (n_points, 2)

    Returns
    -------
    Tuple[List[np.ndarray], Dict[str, Dict[str, float]]]
        - Normalized input data in same structure as training_x
        - Dictionary containing mean and std for each feature:
          {
            "bc_top": {"avg": float, "std": float},
            "bc_bottom": {"avg": float, "std": float},
            "bc_right": {"avg": float, "std": float},
            "bc_left": {"avg": float, "std": float},
            "x_train": {"avg": float, "std": float},
            "y_train": {"avg": float, "std": float}
          }

    Examples
    --------
    >>> raw_data = [
    ...     np.array([300, 310, 305]),  # Top BC
    ...     np.array([290, 295, 292]),  # Bottom BC
    ...     np.array([302, 304]),       # Right BC
    ...     np.array([298, 296]),       # Left BC
    ...     np.array([[0, 0], [1, 0.5], [2, 1]])  # Coordinates
    ... ]
    >>> norm_data, stats = Normalizer(raw_data)
    """

    #Storing the informations
    statistics = {
        "bc_top":{'avg':0, 'std':0},
        "bc_bottom":{'avg':0, 'std':0},
        "bc_right":{'avg':0, 'std':0},
        "bc_left":{'avg':0, 'std':0},
        "x_train":{'avg':0, 'std':0},
        "y_train":{'avg':0, 'std':0}
        }

    #==Seperating the training inputs==
    bc_top_train = np.array(training_x[0])        #Top boundary conditions
    bc_bottom_train = np.array(training_x[1])     #Bottom boundary conditions
    bc_right_train = np.array(training_x[2])      #Right boundary conditions
    bc_left_train = np.array(training_x[3])       #Left boundary conditions

    x_train = np.array(training_x[4][:,0])            #x coordinate
    y_train = np.array(training_x[4][:,1])            #y coordinate

    #Finding the average of the each set
    statistics['bc_top']['avg'] = np.average(bc_top_train)
    statistics['bc_bottom']['avg'] = np.average(bc_bottom_train)
    statistics['bc_right']['avg'] = np.average(bc_right_train)
    statistics['bc_left']['avg'] = np.average(bc_left_train)

    statistics['x_train']['avg'] = np.average(x_train)
    statistics['y_train']['avg'] = np.average(y_train)

    #Finding the standard dev 
    statistics['bc_top']['std'] = np.std(bc_top_train)
    statistics['bc_bottom']['std'] = np.std(bc_bottom_train)
    statistics['bc_right']['std'] = np.std(bc_right_train)
    statistics['bc_left']['std'] = np.std(bc_left_train)

    statistics['x_train']['std'] = np.std(x_train)
    statistics['y_train']['std'] = np.std(y_train)

    #Normalizing the sets
    bc_top_train = (bc_top_train - statistics['bc_top']['avg'])/statistics['bc_top']['std']
    bc_bottom_train = (bc_bottom_train - statistics['bc_bottom']['avg'])/statistics['bc_bottom']['std']
    bc_right_train = (bc_right_train - statistics['bc_right']['avg'])/statistics['bc_right']['std']
    bc_left_train = (bc_left_train - statistics['bc_left']['avg'])/statistics['bc_left']['std']

    x_train = (x_train - statistics['x_train']['avg'])/statistics['x_train']['std']
    y_train = (y_train - statistics['y_train']['avg'])/statistics['y_train']['std']

    coordinates = np.vstack([x_train, y_train])
    coordinates = coordinates.reshape((int(coordinates.size/2) , 2))

    #Reconstructing the training_x(Input array)
    training_x = [bc_top_train, bc_bottom_train, bc_right_train, bc_left_train, coordinates]

    return training_x, statistics
    

def Normalizer(
    x_set: list[list[float]| tuple[np.ndarray, np.ndarray]],
    y_set: list[float]| np.ndarray,
    T_min: float,
    T_max: float,
    length: float,
    width: float
) -> tuple[list[np.ndarray]| np.ndarray]:
    """
    Normalizes boundary conditions and coordinates for heat transfer problems.
    
    Normalizes all inputs to the range [0, 1] using min-max scaling relative to:
    - Temperature bounds (T_min, T_max) for boundary conditions and outputs
    - Domain dimensions (length, width) for spatial coordinates

    Parameters
    ----------
    x_set : List[Union[List[float], Tuple[np.ndarray, np.ndarray]]]
        Input data containing:
        [0] Top boundary temperatures
        [1] Bottom boundary temperatures  
        [2] Right boundary temperatures
        [3] Left boundary temperatures
        [4] Tuple of (x_coordinates, y_coordinates)
    y_set : Union[List[float], np.ndarray]
        Target temperature values to be standardized
    T_min : float
        Minimum temperature value in original scale (used for normalization)
    T_max : float 
        Maximum temperature value in original scale (used for normalization)
    length : float
        Domain length in x-direction (for coordinate normalization)
    width : float
        Domain width in y-direction (for coordinate normalization)

    Returns
    -------
    Tuple[List[np.ndarray], np.ndarray]
        - First element: List of standardized inputs in same order as x_set:
          [top_bc, bottom_bc, right_bc, left_bc, coordinates]
          where coordinates is a numpy array of shape (n_points, 2)
        - Second element: Standardized temperature values (y_set)

    Examples
    --------
    >>> x_data = [
    ...     [300, 310, 305],  # Top BC
    ...     [290, 295, 292],  # Bottom BC
    ...     [302, 304],       # Right BC
    ...     [298, 296],       # Left BC
    ...     (np.array([0, 1, 2]), np.array([0, 0.5, 1]))  # Coordinates
    ... ]
    >>> y_data = [350, 340, 335]
    >>> x_std, y_std = Standardizer(x_data, y_data, 290, 350, 2.0, 1.0)
    """

    #==Seperating the training inputs==
    bc_top = (np.array(x_set[0]) - T_min)/(T_max - T_min)        #Top boundary conditions
    bc_bottom = (np.array(x_set[1]) - T_min)/(T_max - T_min)     #Bottom boundary conditions
    bc_right = (np.array(x_set[2]) - T_min)/(T_max - T_min)      #Right boundary conditions
    bc_left = (np.array(x_set[3]) - T_min)/(T_max - T_min)       #Left boundary conditions

    x_location = np.array(x_set[4][:,0])/(length+width)                     #x coordinate
    y_location = np.array(x_set[4][:,1])/(length+width)                     #y coordinate

    y_set_standardized =  (np.array(y_set) - T_min)/(T_max - T_min)           #Temperatures

    coordinates = np.vstack([x_location, y_location])
    coordinates = coordinates.reshape((int(coordinates.size/2) , 2))

    #Reconstructing the x_set(Input array)
    x_set_standardized = [bc_top, bc_bottom, bc_right, bc_left, coordinates]

    return x_set_standardized, y_set_standardized


#___________________________________________________________________________
#Training Loop for the generalizer  model
#___________________________________________________________________________

def TrainGeneralizer(model:keras.Model, 
          training_x:list[np.ndarray], 
          training_y:np.ndarray, 
          validation_x:np.ndarray,
          validation_y:np.ndarray,
          batch_size:int=128,
          epochs:int=1,
          loss_func:keras.losses.Loss=tf.keras.losses.MeanAbsoluteError()):
    """
    Train a physics-informed neural network (PINN) with combined data and physics loss.

    Parameters
    ----------
    model : keras.Model
        Compiled Keras model to be trained. Must be compiled with an optimizer.
    training_x : list of np.ndarray
        List containing boundary conditions and coordinates in order:
        [bc_top, bc_bottom, bc_right, bc_left, coordinates].
    training_y : np.ndarray
        Array of target temperature values for training.
    validation_x : np.ndarray
        Validation input data (same structure as training_x).
    validation_y : np.ndarray
        Validation target values.
    batch_size : int, optional
        Number of samples per gradient update. Set to None for full-batch training.
        Default is 128.
    epochs : int, optional
        Number of epochs to train. Default is 1.
    loss_func : keras.losses.Loss, optional
        Loss function for data term. Default is MeanAbsoluteError().

    Returns
    -------
    history : dict
        Dictionary containing training metrics with keys:
        - 'physical_loss' : list of physics (PDE) loss values per epoch
        - 'data_loss' : list of data loss values per epoch
        - 'total_loss' : list of combined loss values per epoch
        - 'validation_loss' : list of validation losses per epoch

    Notes
    -----
    - Implements custom batch generation through helper function __batchMaker
    - Uses GradientTape for manual gradient calculation
    - Includes automatic random shuffling of training data each epoch
    - Implements naive early stopping when validation loss stagnates
    - Prints detailed loss metrics after each epoch

    Examples
    --------
    >>> # Training settings
    >>> num_epochs = 20
    >>> batch_size = 256
    >>>
    >>> # Compile model with optimizer
    >>> model_generalizer.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001))
    >>>
    >>> # Run training
    >>> history = TrainGeneralizer(
    ...     model=model_generalizer,
    ...     training_x=x_training,
    ...     training_y=y_training,
    ...     validation_x=x_testing,
    ...     validation_y=y_testing,
    ...     batch_size=batch_size,
    ...     epochs=num_epochs
    ... )
    >>> 
    >>> # Access training history
    >>> print(history['validation_loss'])
    """

    #Calculating the number of required batches
    num_batches = ceil(len(training_x[0]) / batch_size)
    
    #Storing history
    history = {
        'physical_loss':[],
        'data_loss':[],
        'total_loss':[],
        'validation_loss':[]
    }

    #==Training loop==
    for epoch in range(epochs):
        #Shuffeling the training data
        training_x, training_y = __randomize(training_x = training_x, training_y = training_y)

        #==Storing the losses for logs==
        #Storing the losses 
        physical_losses = []
        data_losses = []
        total_losses = []

        for batch_count in range(0,num_batches):
            #Selecting the data for the batch
            bc_top, bc_bottom, bc_right, bc_left, x, y, T_true = __batchMaker(training_x, training_y, batch_count, num_batches, batch_size)

            with tf.GradientTape() as tape:  
                #Calculating the loss
                total_loss, pde_loss, data_loss = LossFunction(model, loss_func ,bc_top, bc_bottom, bc_right, bc_left, x, y, T_true)
                
            #==Optimizing the model==
            gradients = tape.gradient(total_loss, model.trainable_variables)
            model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            #Storing the data_loss
            # total_losses.append(total_loss)

            #Storing the losses and predictions for logging
            physical_losses.extend(pde_loss.numpy())
            data_losses.append(data_loss.numpy())
            total_losses.append(total_loss.numpy())

        #==Logging===
        print("-----------------")
        print(f"Epoch : {epoch+1}/{epochs}")
        print(f"Physical loss: {np.mean(physical_losses)}")
        print(f"Data loss: {np.mean(data_losses)}")
        print(f"Total loss: {np.mean(total_losses)}")

        #Validation accuracy
        validation_prediction = model.predict(validation_x)
        validation_loss = loss_func(validation_prediction, validation_y)
        print(f"Validation loss: {validation_loss.numpy()}")

        #Stroing the data to history
        history['physical_loss'].append(np.mean(physical_losses))
        history['data_loss'].append(np.mean(data_loss))
        history['total_loss'].append(np.mean(total_loss))
        history['validation_loss'].append(validation_loss.numpy())

 
        #==Naive early stopping==
        if __has_loss_stagnated(history['validation_loss']):
            return history


    return history


#Calculating if the training has stalled or not
def __has_loss_stagnated(validation_loss: np.ndarray, window: int = 6) -> bool:
    
    if len(validation_loss) < window:
        return False                                            # Not enough data
    
    recent_losses = validation_loss[-window:]
    
    #==Finding the slope of the data
    no_downward_trend = (
        np.polyfit(range(window), recent_losses, 1)[0] > -0.15/(3300-100)   # Flat/sloping up
    )
    
    return no_downward_trend

         
#Shuffling the dataset
def __randomize(training_x, training_y):

    #==Seperating the training inputs==
    bc_top_train = training_x[0]        #Top boundary conditions
    bc_bottom_train = training_x[1]     #Bottom boundary conditions
    bc_right_train = training_x[2]      #Right boundary conditions
    bc_left_train = training_x[3]       #Left boundary conditions
    coords_train = training_x[4]        #Coordinates
    x_train = coords_train[:,0]         #x coordinate
    y_train = coords_train[:,1]         #y coordinate

    #Get number of samples
    num_samples = len(x_train)
    
    #Generate random permutation
    shuffled_indices = np.random.permutation(num_samples)
    
    #Shuffle each array
    bc_top_train = bc_top_train[shuffled_indices]
    bc_bottom_train = bc_bottom_train[shuffled_indices]
    bc_right_train = bc_right_train[shuffled_indices]
    bc_left_train = bc_left_train[shuffled_indices]
    x_train = x_train[shuffled_indices]
    y_train = y_train[shuffled_indices]
    
    #Rebuild coordinates array
    coords_train = np.column_stack((x_train, y_train))
    
    #Rebuild training_x
    shuffled_training_x = [
        bc_top_train,
        bc_bottom_train,
        bc_right_train,
        bc_left_train,
        coords_train
    ]
    
    #Shuffleing the targets
    shuffled_training_y = training_y[shuffled_indices]
    
    return shuffled_training_x, shuffled_training_y


#Will return the data for each batch 
def __batchMaker(training_x, training_y, batch_count, num_batches, batch_size):

    #==Seperating the training inputs==
    bc_top_train = training_x[0]        #Top boundary conditions
    bc_bottom_train = training_x[1]     #Bottom boundary conditions
    bc_right_train = training_x[2]      #Right boundary conditions
    bc_left_train = training_x[3]       #Left boundary conditions

    x_train = training_x[4][:,0]             #x coordinate
    y_train = training_x[4][:,1]             #y coordinate

    #For the final batch
    if batch_count == num_batches - 1:
        bc_top = bc_top_train[batch_count*batch_size:]
        bc_bottom = bc_bottom_train[batch_count*batch_size:]
        bc_right = bc_right_train[batch_count*batch_size:]
        bc_left = bc_left_train[batch_count*batch_size:]

        x = tf.convert_to_tensor(x_train[batch_count*batch_size:])
        y = tf.convert_to_tensor(y_train[batch_count*batch_size:])
        
        T_true = training_y[batch_count*batch_size:]

    else:
        bc_top = bc_top_train[batch_count*batch_size:(batch_count+1)*batch_size]
        bc_bottom = bc_bottom_train[batch_count*batch_size:(batch_count+1)*batch_size]
        bc_right = bc_right_train[batch_count*batch_size:(batch_count+1)*batch_size]
        bc_left = bc_left_train[batch_count*batch_size:(batch_count+1)*batch_size]

        x = tf.convert_to_tensor(x_train[batch_count*batch_size:(batch_count+1)*batch_size])
        y = tf.convert_to_tensor(y_train[batch_count*batch_size:(batch_count+1)*batch_size])

        T_true = training_y[batch_count*batch_size:(batch_count+1)*batch_size]

        #For debuging perpouses
        # print('-----------------')
        # print(batch_size)
        # print(bc_top.shape)
        # print(bc_bottom.shape)
        # print(bc_right.shape)
        # print(bc_left.shape)
        # print(mesh_size.shape)
        # print(T_true)
        # print(x)
        # print(y)

    return bc_top, bc_bottom, bc_right, bc_left, x, y, T_true


#The loss function for the generalizer model
@tf.function
def LossFunction(model, loss_func, bc_top, bc_bottom, bc_right, bc_left, x, y, T_true):
    """
    Compute combined loss for physics-informed neural network (PINN) generalizer model.

    Calculates:
    1. Physics loss (Laplace equation ∇²T = 0 residual)
    2. Data matching loss (difference between predictions and ground truth)
    3. Weighted total loss (default 10:1 physics-to-data weighting)

    Parameters
    ----------
    model : keras.Model
        PINN model that takes boundary conditions and coordinates as input:
        [bc_top, bc_bottom, bc_right, bc_left, coordinates]
    loss_func : keras.losses.Loss
        Loss function for data term (e.g., MAE, MSE)
    bc_top : array-like
        Top boundary condition values
    bc_bottom : array-like
        Bottom boundary condition values
    bc_right : array-like
        Right boundary condition values
    bc_left : array-like
        Left boundary condition values
    x : tf.Tensor
        X-coordinates for evaluation points
    y : tf.Tensor
        Y-coordinates for evaluation points
    T_true : tf.Tensor
        Ground truth temperature values at (x,y) locations

    Returns
    -------
    tuple
        Three-element tuple containing:
        - total_loss : Weighted sum of physics and data losses (10:1 weighting)
        - pde_loss : Vector of Laplace equation residuals (∇²T) at each point
        - data_loss : Data mismatch scalar value

    Notes
    -----
    - Uses nested GradientTape for second derivative calculations:
        1. Inner tape computes first derivatives (∂T/∂x, ∂T/∂y)
        2. Outer tape computes second derivatives (∂²T/∂x², ∂²T/∂y²)
    - Implements hard-coded 10:1 weighting (physics:data terms)
    - Explicitly cleans tapes to manage memory
    - Decorated with @tf.function for graph execution efficiency

    Examples
    --------
    >>> # Using MSE for data loss
    >>> total_loss, pde_residuals, data_error = LossFunction(
    ...     model=model_generalizer,
    ...     loss_func=tf.keras.losses.MeanSquaredError(),
    ...     bc_top=top_bc_values,
    ...     bc_bottom=bottom_bc_values,
    ...     bc_right=right_bc_values,
    ...     bc_left=left_bc_values,
    ...     x=x_coords,
    ...     y=y_coords,
    ...     T_true=reference_temps
    ... )
    >>>
    >>> # To modify loss weighting (change line in function):
    >>> # total_loss = data_loss + weight * physical_loss

    Implementation Details
    ---------------------
    1. Converts coordinates to tensor and reshapes to (N, 2) format
    2. Computes temperature predictions using all boundary conditions
    3. Calculates PDE residuals using finite differences via autograd
    4. Combines losses with physics-dominated weighting (10×)
    """
            
    #The presistent=true is neccesary if higher order derivatives are to be calculated -> del tape must be added
    with tf.GradientTape(persistent=True) as first_tape:   
        #Adding x, y to the tape
        first_tape.watch(x)
        first_tape.watch(y)

        #For the calculation of the second derivative
        with tf.GradientTape(persistent=True) as second_tape: 
            #Adding x, y to the tape
            second_tape.watch(x)
            second_tape.watch(y)  

            loc_vec = tf.convert_to_tensor([x,y])
            loc_vec = tf.reshape(loc_vec, (len(x), 2))

            #Making predictions
            T_pred = model([tf.convert_to_tensor(bc_top), tf.convert_to_tensor(bc_bottom), tf.convert_to_tensor(bc_right), tf.convert_to_tensor(bc_left), loc_vec])

        #Calculating the first derivatives
        dT_dx = second_tape.gradient(T_pred, x)
        dT_dy = second_tape.gradient(T_pred, y)

        #Calculating the second derivative
        d2T_dx2 = first_tape.gradient(dT_dx, x)
        d2T_dy2 = first_tape.gradient(dT_dy, y)

        #==Calculating the loss==
        #🪀Physcial loss ∇²T = 0
        pde_loss = tf.math.abs(d2T_dx2 + d2T_dy2)                                   #Is a vector containing values for each sample
        physical_loss = loss_func(pde_loss, tf.zeros_like(pde_loss))

        #🗞️Data loss
        data_loss = loss_func(T_true, T_pred)
        # data_loss = tf.math.reduce_mean(tf.math.sqrt((T_pred - T_true)**2))

        #👇🏻This line can be modified to add weight to the losses
        total_loss =  data_loss + 10 * physical_loss
        # total_loss = data_loss
        
    #Cleaning the tape for lowering the memory consumption
    del first_tape
    del second_tape

    return total_loss, pde_loss, data_loss


#___________________________________________________________________________
# Training loop for solver model
#___________________________________________________________________________

def TrainSolver(model:keras.Model, 
                training_x:ThermalBoundaryConditions, 
                validation_data:list[np.ndarray,np.ndarray,np.ndarray],                
                batch_size:int=128,
                epochs:int=1,
                loss_func:keras.losses.Loss=tf.keras.losses.MeanAbsoluteError()):
    """
    Train a physics-informed neural network (PINN) solver for thermal problems with combined PDE and boundary condition losses.

    The training process involves:
    1. Boundary condition enforcement at specified locations
    2. PDE satisfaction at random interior points
    3. Validation against reference solutions

    Parameters
    ----------
    model : keras.Model
        Compiled Keras model with optimizer configured. The model should accept
        (x,y) coordinate inputs and output temperature predictions.
    training_x : ThermalBoundaryConditions
        Object containing domain specifications and boundary conditions.
        Should include:
        - Domain dimensions (length, width)
        - Boundary temperatures (top, bottom, left, right)
        - Mesh parameters (dx, dy)
    validation_data : list of np.ndarray
        Validation dataset in format [x_coords, y_coords, reference_temperatures].
        Arrays should have matching shapes.
    batch_size : int, optional
        Number of boundary points per training batch. Default is 128.
    epochs : int, optional
        Number of complete passes through the training data. Default is 1.
    loss_func : keras.losses.Loss, optional
        Loss function for boundary condition matching. Default is MAE.

    Returns
    -------
    dict
        Training history dictionary with keys:
        - 'physical_loss' : PDE residual magnitudes
        - 'bc_loss' : Boundary condition mismatch
        - 'total_loss' : Combined loss
        - 'validation_loss' : Reference solution comparison

    Examples
    --------
    >>> # Configure model with AdamW optimizer
    >>> model_solver.compile(optimizer=tf.keras.optimizers.AdamW())
    >>>
    >>> # Run training with boundary conditions and validation data
    >>> history = TrainSolver(
    ...     model=model_solver,
    ...     training_x=conditions,  # ThermalBoundaryConditions object
    ...     validation_data=validation_data,  # [x_valid, y_valid, T_valid]
    ...     epochs=70,
    ...     batch_size=128
    ... )
    >>>
    >>> # Example of accessing training results:
    >>> final_pde_loss = history['physical_loss'][-1]
    >>> final_validation_error = history['validation_loss'][-1]

    Notes
    -----
    - Implements dual optimization:
      1. Boundary term optimization with structured batches
      2. PDE term optimization with random interior sampling
    - Automatically shuffles boundary points each epoch
    - Prints progressive loss metrics during training
    - Uses 10 random interior points per boundary batch (hardcoded)
    """

    #Shuffeling the training data
    tempeatures ,coordiantes_x, coordiantes_y = __dataOrganizer(training_x)

    #Calculating the number of required batches
    num_batches = ceil(len(tempeatures) / batch_size) 

    #Getting the valdiation data organized
    x_valid = validation_data[0]
    y_valid = validation_data[1]
    T_valid = validation_data[2]
    
    #Storing history
    history = {
        'physical_loss':[],
        'bc_loss':[],               #Boundary condition loss
        'total_loss':[],
        'validation_loss':[]
    }

    #==Training loop==
    for epoch in range(epochs):
        #Shuffeling the training data
        tempeatures ,coordiantes_x, coordiantes_y = __dataOrganizer(training_x)

        #==Storing the losses for logs==
        #Storing the losses 
        physical_losses = []
        bc_losses = []
        total_losses = []

        for batch_count in range(0,num_batches):
            #Selecting the data for the batch
            x, y, T_true = __batchMakerSolver(coordiantes_x, coordiantes_y, tempeatures, batch_count, num_batches, batch_size)

            #==On the boundary==
            with tf.GradientTape() as tape:  
                #Calculating the loss
                total_loss, pde_loss, data_loss = LossFunctionSolver(model, loss_func, x, y, T_true)

                #<--test-->
                # print("_______DEBUG________")
                # print(f"Physical loss {pde_loss}")
                # print(f"total loss {total_loss}")
                # print(f"bc_loss loss {data_loss}")
                
            #==Optimizing the model==
            gradients = tape.gradient(total_loss, model.trainable_variables)
            model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            #=Inside the domain==
            with tf.GradientTape() as tape:
                #Random locations
                n = 10 
                x = np.random.uniform(low=0.0, high=training_x.length, size=x.size) 
                y = np.random.uniform(low=0.0, high=training_x.width, size=y.size)

                #Calculating the loss(The last parameter of the function is not important)
                _, pde_loss_inisde, _ = LossFunctionSolver(model, loss_func, x, y, y)

            #==Optimizing the model==
            gradients = tape.gradient(pde_loss_inisde, model.trainable_variables)
            model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))


            #<--test-->
            # return gradients
            #<--test-->

            #Storing the data_loss
            # total_losses.append(total_loss)

            #Storing the losses and predictions for logging
            physical_losses.extend(pde_loss.numpy())
            bc_losses.append(data_loss.numpy())
            total_losses.append(total_loss.numpy())

        #==Logging===
        print("-----------------")
        print(f"Epoch : {epoch+1}/{epochs}")
        print(f"Physical loss: {np.mean(physical_losses)}")
        print(f"Boundary loss: {np.mean(bc_losses)}")
        print(f"Total loss: {np.mean(total_losses)}")

        #Validation accuracy
        validation_prediction = predict_temperature(model, x_valid, y_valid)
        validation_loss = loss_func(validation_prediction, T_valid)
        print(f"Validation loss: {validation_loss.numpy()}")

        #Stroing the data to history
        history['physical_loss'].append(np.mean(physical_losses))
        history['bc_loss'].append(np.mean(data_loss))
        history['total_loss'].append(np.mean(total_loss))
        history['validation_loss'].append(validation_loss.numpy())

    return history

#The loss function for the solver model
@tf.function
def LossFunctionSolver(model, loss_func, x, y, T_true):
    """
    Compute combined loss for physics-informed neural network (PINN) thermal solver.

    Calculates:
    1. Physics loss (Laplace equation ∇²T = 0 residual)
    2. Boundary condition/data matching loss
    3. Weighted total loss (default 10:1 physics-to-data weighting)

    Parameters
    ----------
    model : keras.Model
        PINN model that takes (x,y) coordinates and outputs temperature predictions
    loss_func : keras.losses.Loss
        Loss function for boundary condition term (e.g., MAE, MSE)
    x : tf.Tensor
        X-coordinates for evaluation (1D array)
    y : tf.Tensor
        Y-coordinates for evaluation (1D array)
    T_true : tf.Tensor
        Reference temperatures at (x,y) locations

    Returns
    -------
    tuple
        Three-element tuple containing:
        - total_loss : Weighted sum of physics and BC losses
        - pde_loss : Vector of Laplace equation residuals (∇²T) at each point
        - bc_loss : Boundary condition mismatch scalar value

    Notes
    -----
    - Uses nested GradientTape for second derivative calculations
    - Implements hard-coded 10:1 weighting (physics:BC terms)
    - Cleans tapes explicitly to manage memory
    - Decorated with @tf.function for graph execution

    Examples
    --------
    >>> # Using MAE for boundary conditions
    >>> total_loss, pde_residuals, bc_error = LossFunctionSolver(
    ...     model=model_solver,
    ...     loss_func=tf.keras.losses.MeanAbsoluteError(),
    ...     x=x_coords,
    ...     y=y_coords,
    ...     T_true=reference_temps
    ... )
    >>>
    >>> # Custom weighting (modify line in function):
    >>> # total_loss = bc_loss + physical_weight * physical_loss

    Implementation Details
    ---------------------
    1. First derivatives (∂T/∂x, ∂T/∂y) computed via inner tape
    2. Second derivatives (∂²T/∂x², ∂²T/∂y²) computed via outer tape
    3. Physics loss: L2 norm of ∇²T residuals
    4. BC loss: Specified loss_func between predictions and truths
    """
            
    #The presistent=true is neccesary if higher order derivatives are to be calculated -> del tape must be added
    with tf.GradientTape(persistent=True) as first_tape:   
        #Adding x, y to the tape
        first_tape.watch(x)
        first_tape.watch(y)

        #For the calculation of the second derivative
        with tf.GradientTape(persistent=True) as second_tape: 
            #Adding x, y to the tape
            second_tape.watch(x)
            second_tape.watch(y)  

            loc_vec = tf.convert_to_tensor([x,y])
            loc_vec = tf.reshape(loc_vec, (len(x), 2))

            #Making predictions
            T_pred = model([x, y])

        #Calculating the first derivatives
        dT_dx = second_tape.gradient(T_pred, x)
        dT_dy = second_tape.gradient(T_pred, y)

        #Calculating the second derivative
        d2T_dx2 = first_tape.gradient(dT_dx, x)
        d2T_dy2 = first_tape.gradient(dT_dy, y)

        #==Calculating the loss==
        #🪀Physcial loss ∇²T = 0
        pde_loss = tf.math.abs(d2T_dx2 + d2T_dy2)                                   #Is a vector containing values for each sample
        physical_loss = loss_func(pde_loss, tf.zeros_like(pde_loss))

        #🗞️Data loss
        bc_loss = loss_func(T_true, T_pred)
        # data_loss = tf.math.reduce_mean(tf.math.sqrt((T_pred - T_true)**2))

        #👇🏻This line can be modified to add weight to the losses
        # total_loss =  bc_loss
        total_loss = bc_loss + 10 * physical_loss
        
    #Cleaning the tape for lowering the memory consumption
    del first_tape
    del second_tape

    return total_loss, pde_loss, bc_loss


#Creates batches of coordinate-temperature pairs for neural network training.
def __batchMakerSolver(coordinates_x:np.ndarray, 
                       coordinates_y:np.ndarray,
                       temperatures:np.ndarray, 
                       batch_count:int, 
                       num_batches:int, 
                       batch_size:int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    #For the final batch
    if batch_count == num_batches - 1:
        
        #Inputs of the network
        x = coordinates_x[batch_count*batch_size:]
        y = coordinates_y[batch_count*batch_size:]
        
        #Output of the network
        T_true = temperatures[batch_count*batch_size:]

    else:

        x = coordinates_x[batch_count*batch_size:(batch_count+1)*batch_size]
        y = coordinates_y[batch_count*batch_size:(batch_count+1)*batch_size]

        T_true = temperatures[batch_count*batch_size:(batch_count+1)*batch_size]

        #For debuging perpouses
        # print('-----------------')
        # print(T_true)
        # print(x)
        # print(y)

    return x ,y, T_true

#Organizes boundary condition data into shuffled coordinate-temperature pairs for PINN training.
def __dataOrganizer(training_x:ThermalBoundaryConditions) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    #==Extracting the data==
    #Boundary conditions
    bc_top = training_x.T_top
    bc_bottom = training_x.T_bottom
    bc_right= training_x.T_right
    bc_left = training_x.T_left

    #Mesh size
    dx = training_x.dx
    dy = training_x.dy

    #Geometry
    length = training_x.length
    width = training_x.width

    #Finding the coordiantes of each set of points
    coordinates_bc_bottom = np.linspace(1,len(bc_bottom), len(bc_bottom)) * dx
    coordinates_bc_bottom = np.vstack([coordinates_bc_bottom, np.zeros_like(coordinates_bc_bottom)])

    coordinates_bc_left = np.linspace(1,len(bc_left), len(bc_left)) * dy
    coordinates_bc_left = np.vstack([np.zeros_like(coordinates_bc_left), coordinates_bc_left])

    coordinates_bc_right = np.linspace(1,len(bc_right), len(bc_right)) * dy
    coordinates_bc_right = np.vstack([np.ones_like(coordinates_bc_right) * length, coordinates_bc_right])

    coordinates_bc_top = np.linspace(1,len(bc_top), len(bc_top)) * dx
    coordinates_bc_top = np.vstack([coordinates_bc_top, np.ones_like(coordinates_bc_top) * width])

    #==Vectorize the dataset
    temperatures = np.hstack([bc_top, bc_bottom, bc_right, bc_left])
    coordinates_x = np.hstack([coordinates_bc_top[0], coordinates_bc_bottom[0], coordinates_bc_right[0], coordinates_bc_left[0]])
    coordinates_y = np.hstack([coordinates_bc_top[1], coordinates_bc_bottom[1], coordinates_bc_right[1], coordinates_bc_left[1]])


    # #==Shuffle the dataset==

    #Generate random permutation
    shuffled_indices = np.random.permutation(temperatures.size)

    #Shuffling the arrays
    temperatures = temperatures[shuffled_indices]
    coordinates_x = coordinates_x[shuffled_indices]
    coordinates_y = coordinates_y[shuffled_indices]

    #<--test-->
    # print(f"1>{temperatures.shape}")
    # print(f"2>{coordinates_x.shape}")
    # print(f"3>{coordinates_y.shape}")
    #<--test-->

    return temperatures, coordinates_x, coordinates_y


def GenerateTemperatureField(model: keras.Model, conditions: ThermalBoundaryConditions) -> np.ndarray:
    """
    Generates a temperature field prediction on a uniform grid using a trained PINN model.

    Computes temperature values at all grid points defined by the conditions object,
    returning results in row-major order from bottom-left to top-right.

    Parameters
    ----------
    model : keras.Model
        Trained physics-informed neural network with:
        - Input: List of two arrays [x_coords, y_coords], each shaped (N, 1)
        - Output: Temperature predictions shaped (N,)
    conditions : ThermalBoundaryConditions
        Boundary conditions container with attributes:
        - length : float
            Domain length in x-direction (meters)
        - width : float
            Domain width in y-direction (meters)
        - dx : float
            Grid spacing in x-direction (meters)
        - dy : float
            Grid spacing in y-direction (meters)

    Returns
    -------
    np.ndarray
        1D array of temperature values in °C, ordered in row-major (C-style) sequence:
        [T(0,0), T(dx,0), T(2dx,0), ..., T(0,dy), T(dx,dy), ...]
        Shape: (N,) where N = (width/dy + 1) * (length/dx + 1)

    Raises
    ------
    ValueError
        If model prediction fails or grid generation parameters are invalid

    Examples
    --------
    >>> from FDM import ThermalBoundaryConditions
    >>> from pinnTools import BuildFunc
    >>> import matplotlib.pyplot as plt

    >>> # Define domain and boundary conditions
    >>> conditions = ThermalBoundaryConditions(
    ...     length=60, width=60, dx=0.25, dy=0.25,
    ...     T_top=239*[500], T_bottom=239*[400],
    ...     T_left=239*[100], T_right=239*[700]
    ... )

    >>> # Build and use model
    >>> model = BuildFunc(num_hidden_layers=10, num_units=64,
    ...                  input_shapes=[1, 1])
    >>> temperatures = GenerateTemperatureField(model, conditions)

    >>> # Visualize results
    >>> grid_shape = (int(conditions.width/conditions.dy) + 1,
    ...               int(conditions.length/conditions.dx) + 1)
    >>> plt.imshow(temperatures.reshape(grid_shape),
    ...            extent=[0,60,0,60], origin='lower')
    >>> plt.colorbar(label='Temperature (°C)')
    >>> plt.show()

    Notes
    -----
    - Grid generation includes both endpoints (0 and length/width)
    - Uses numpy.meshgrid with indexing='xy' (Cartesian convention)
    - For visualization, reshape output using:
        temps_2d = temperatures.reshape(
            int(conditions.width/conditions.dy) + 1,
            int(conditions.length/conditions.dx) + 1
        )
    """
    length = conditions.length
    width = conditions.width
    dx = conditions.dx
    dy = conditions.dy

    # Create grid points
    x_points = np.arange(dx, length + dx/2 -dx, dx)
    y_points = np.arange(dy, width + dy/2 - dy, dy)
    xx, yy = np.meshgrid(x_points, y_points)

    # Vectorize coordinates
    x_coords = xx.reshape(-1, 1)
    y_coords = yy.reshape(-1, 1)
    
    # Predict temperatures
    temperatures = model.predict([x_coords, y_coords], verbose=0).flatten()
    
    return temperatures


def predict_temperature(model:keras.Model, x:np.ndarray, y:np.ndarray):
    """
    Predict temperature at given coordinates
    
    Parameters
    ----------
    model : keras.Model
        Trained model
    x : float or array-like
        x-coordinate(s)
    y : float or array-like
        y-coordinate(s)
        
    Returns
    -------
    np.ndarray
        Predicted temperature(s)
    """
    # Ensure proper input shape (batch_size, features)
    x_arr = np.array(x).reshape(-1, 1).astype('float32')
    y_arr = np.array(y).reshape(-1, 1).astype('float32')
    
    return model.predict([x_arr, y_arr])


def ValidationDataOrganizer(temperatures: np.ndarray, conditions: ThermalBoundaryConditions) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates a validation dataset of coordinates and corresponding temperature values.

    Creates coordinate pairs (x,y) for each temperature value in the input array,
    mapping array indices to physical coordinates based on the domain specifications
    in the ThermalBoundaryConditions object.

    Parameters
    ----------
    temperatures : np.ndarray
        1D array of temperature values to be validated. The length of this array
        should match the total number of grid points in the domain.
    conditions : ThermalBoundaryConditions
        Container object with domain specifications including:
        - length : float
            Domain length in x-direction (meters)
        - width : float
            Domain width in y-direction (meters)
        - dx : float
            Grid spacing in x-direction (meters)
        - dy : float
            Grid spacing in y-direction (meters)

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        A tuple containing three 1D numpy arrays:
        - x_coord: x-coordinates in physical units (meters)
        - y_coord: y-coordinates in physical units (meters)
        - temperatures: The input temperature array (unchanged)

    Notes
    -----
    - The function assumes temperature values are ordered in row-major (C-style) order,
      corresponding to the meshgrid convention used in GenerateTemperatureField.
    - Coordinate conversion is handled by the private __var_ind_to_xy_coord function.

    Examples
    --------
    >>> from FDM import ThermalBoundaryConditions
    >>> conditions = ThermalBoundaryConditions(
    ...     length=60, width=60, dx=0.25, dy=0.25,
    ...     T_top=239*[500], T_bottom=239*[400],
    ...     T_left=239*[100], T_right=239*[700]
    ... )
    >>> # Simulated temperature field (240x240 grid)
    >>> temps, _, _ =  Solver(conditions) 
    >>> x, y, t = ValidationSet(temps, conditions)
    """
    # Getting the coordinates
    x_coord, y_coord = __var_ind_to_xy_coord(np.linspace(0, len(temperatures), len(temperatures)), conditions)

    return x_coord, y_coord, temperatures


#Will convert the index of the node in the solution vector to the x and y index of the node in the square domain. 
def __var_ind_to_xy_coord(index:int|np.ndarray, conditions:ThermalBoundaryConditions) -> tuple[np.ndarray,np.ndarray]:

    #Finding the number of columns
    dx = conditions.dx
    dy = conditions.dy
    x_vars = int(conditions.length / dx) - 1 

    #Calculating the grid count
    x = index % x_vars
    y = index // x_vars

    #Calculating the position
    x = x*dx + dx
    y = y*dy + dy

    return x, y