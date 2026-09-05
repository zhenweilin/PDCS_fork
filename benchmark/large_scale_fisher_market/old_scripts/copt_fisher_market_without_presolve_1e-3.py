import math
import random
import copy
import numpy as np
import h5py
import scipy.io as sio
import scipy.sparse as sp
import os
import argparse
from pathlib import Path

def load_mat_v73(file_path):
    """
    Load MATLAB v7.3 .mat file using h5py
    """
    try:
        with h5py.File(file_path, 'r') as f:
            # Load variables
            A = sp.csc_matrix((np.array(f['A/data']).flatten(),
                             np.array(f['A/ir']).flatten(),
                             np.array(f['A/jc']).flatten()))
            
            # Load other variables and ensure they're properly shaped
            b = np.array(f.get('b')).flatten()
            c = np.array(f.get('c')).flatten()
            bl = np.array(f.get('bl')).flatten()
            bu = np.array(f.get('bu')).flatten()
            
            # Load scalar values
            mGzero = int(np.array(f.get('mGzero')).item())
            mGnonnegative = int(np.array(f.get('mGnonnegative')).item())
            expG = int(np.array(f.get('expG')).item())
            
            return b, A, c, mGzero, mGnonnegative, expG, bl, bu
            
    except Exception as e:
        print(f"Error loading file: {str(e)}")
        raise

def process_single_file(jld2_path):
    """
    Process a single JLD2 file
    """
    # Convert to absolute path and ensure it exists
    jld2_path = os.path.expanduser(jld2_path)
    if not os.path.exists(jld2_path):
        raise FileNotFoundError(f"JLD2 file not found: {jld2_path}")
    
    # Run Julia script to convert JLD2 to MAT
    julia_cmd = f"julia /Users/zhenweilin/pdhg_clp/code/baseline/read_jld2.jl --data_path {jld2_path}"
    ret = os.system(julia_cmd)
    if ret != 0:
        raise RuntimeError(f"Failed to run Julia script with command: {julia_cmd}")
    
    # Load the generated MAT file
    mat_file = "data.mat"  # This matches the output file name in read_jld2.jl
    if not os.path.exists(mat_file):
        raise FileNotFoundError(f"MAT file not found: {mat_file}")
    
    # Load and process the MAT file
    b, A, c, mGzero, mGnonnegative, expG, bl, bu = load_mat_v73(mat_file)
    
    # Run optimization
    run_optimization(b, A, c, mGzero, mGnonnegative, expG, bl, bu)

def run_optimization(b, A, c, mGzero, mGnonnegative, expG, bl, bu):
    """
    Run the COPT optimization with the exact structure from read_jld2.jl
    """
    import coptpy as cp
    from coptpy import COPT
    
    env = cp.Envr()
    model = env.createModel("copt_exp")

    # Add variables with bounds
    x = model.addMVar(A.shape[1], lb=bl, ub=bu)

    # Set objective
    obj = c @ x
    model.setObjective(obj, COPT.MINIMIZE)

    # Add equality constraints (first mGzero rows)
    A_equal = A[:mGzero, :]
    b_equal = b[:mGzero]
    model.addConstrs(A_equal @ x == b_equal)

    # Add exponential cone constraints
    u = model.addMVar(int(expG * 3), lb=-np.inf, ub=np.inf)
    A_exp = A[mGzero:, :]
    b_exp = b[mGzero:]
    model.addConstrs(A_exp @ x - b_exp == u)

    # Add exponential cones (matching the structure in read_jld2.jl)
    for i in range(int(expG)):
        model.addExpCone([u[3*i + 2].item(), u[3*i + 1].item(), u[3*i].item()], 
                        COPT.EXPCONE_PRIMAL)
    model.setParam("Presolve", 0)
    # Solve the model
    model.setParam("FeasTol", 1e-3)
    model.setParam("RelGap", 1e-3)
    model.setParam("DualTol", 1e-3)
    model.setParam("TimeLimit", 3600.0 * 2)
    model.setParam("BarIterLimit", 2**10 - 1)
    model.solve()
    
    # Print solution status and objective value
    print(f"Optimization status: {model.status}")
    if model.status == COPT.OPTIMAL:
        print(f"Optimal objective value: {model.objval}")

def main():
    parser = argparse.ArgumentParser(description='Process Fisher market optimization problem')
    parser.add_argument('--data_path', type=str, required=True,
                      help='Path to a JLD2 file or directory containing JLD2 files')
    
    args = parser.parse_args()
    data_path = os.path.expanduser(args.data_path)
    
    if os.path.isfile(data_path):
        if data_path.endswith('.jld2'):
            print(f"Processing file: {data_path}")
            process_single_file(data_path)
        else:
            print(f"Error: {data_path} is not a JLD2 file")
    elif os.path.isdir(data_path):
        jld2_files = list(Path(data_path).glob('*.jld2'))
        # sort the jld2 files by the first number in the file name
        jld2_files.sort(key=lambda x: int(x.name.split('_')[0]))
        if not jld2_files:
            print(f"No JLD2 files found in {data_path}")
            return
        
        for file in jld2_files:
            print("case begin summary----------------------------")
            print(f"\nProcessing {file}")
            process_single_file(str(file))
            print("case end summary----------------------------")
    else:
        print(f"Error: {data_path} is not a valid file or directory")

if __name__ == "__main__":
    main()