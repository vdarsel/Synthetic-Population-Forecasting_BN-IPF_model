import numpy as np

            
def normalize(data):
    data = np.max([np.zeros_like(data), data],0)
    return (data)/np.sum((data), 0, keepdims=True)


def encoding_coefficient_tanh(coefficients, reference_coefficients):
    coefs = np.array([Y_val.reshape(-1) for Y_val in coefficients])
    coefs_dif = coefs-reference_coefficients
    coefs_to_estimate = np.arctanh(coefs_dif)
    return coefs_to_estimate

def decoding_coefficient_tanh(coefficients, reference_coefficients):
    assert len(coefficients) == np.prod(reference_coefficients.shape)
    # print(coefs)
    cpd_values_shape = reference_coefficients.shape
    coefs_before_transformation = coefficients.reshape(cpd_values_shape)
    
    coefs_before_transformation = np.minimum(coefs_before_transformation, 1)
    coefs_before_transformation = np.maximum(coefs_before_transformation, -1)
    
    coefs_before_softmax = reference_coefficients+np.tanh(coefs_before_transformation)
    
    return normalize(coefs_before_softmax)

def encoding_coefficient_no_embedding(coefficients, reference_coefficients = None):
    return np.array([Y_val.reshape(-1) for Y_val in coefficients])

def decoding_coefficient_no_embedding(coefficients, reference_coefficients):
    assert len(coefficients) == np.prod(reference_coefficients.shape)

    cpd_values_shape = reference_coefficients.shape
    coefs = coefficients.reshape(cpd_values_shape)
    
    coefs = np.minimum(coefs, 1)
    coefs = np.maximum(coefs, 0)

    return normalize(coefs)

def encoding_coefficient_log(coefficients, reference_coefficients = None):
    coefs = np.array([Y_val.reshape(-1) for Y_val in coefficients])
    
    coefs_dif = np.max([coefs,np.ones_like(coefs)*1e-15],0)

    coefs_to_estimate = np.log(coefs_dif)

    return coefs_to_estimate

def decoding_coefficient_log(coefficients, reference_coefficients):
    assert len(coefficients) == np.prod(reference_coefficients.shape)
    # print(coefs)
    cpd_values_shape = reference_coefficients.shape
    coefs = coefficients.reshape(cpd_values_shape)
    
    coefs = np.exp(coefs)
            
    return normalize(coefs)


def encoding_coefficient_log_ratio(coefficients, reference_coefficients):
    dim = len(coefficients[0].shape)
    res = []
    if dim>1:
        reference_coefficients = reference_coefficients.reshape(coefficients[0].shape[0],-1)
    
    arg_max = np.argmax(reference_coefficients,0, keepdims=True)
    
    res = []
    
    for coefs in coefficients:
        if dim>1:
            coefs = coefs.reshape(coefs.shape[0],-1)
        
        maxvals = np.take_along_axis(coefs, arg_max, 0)
        
        coefs = np.maximum(coefs, 1e-15)
        maxvals = np.maximum(maxvals, 1e-15)
        
        res.append((np.log(coefs) - np.log(maxvals)).reshape(-1))

    return np.array(res)


def decoding_coefficient_log_ratio(coefficients, reference_coefficients):
    assert len(coefficients) == np.prod(reference_coefficients.shape)
    
    # print(coefficients)
    cpd_values_shape = reference_coefficients.shape
    coefs = coefficients.reshape(cpd_values_shape)

    coefs = np.exp(coefs)
    
    coefs =  normalize(coefs)
    # print(coefs)
    return coefs
