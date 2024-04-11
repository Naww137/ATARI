import numpy as np

test_xs_total={"Distribution":np.ones(5)}
test_xs_total["Jacobian"]=np.ones((5,5))*1
np.fill_diagonal(test_xs_total["Jacobian"],test_xs_total["Distribution"]*2)

#print(test_xs_total["Distribution"])
#print("")
#print(test_xs_total["Jacobian"])

def send_both():
    return(1,2)

yellow={}

yellow["one"],yellow["two"]=send_both()

red=np.diag([2,2,2,2])
green=np.concatenate([red],1)
#print(green)

a={"a":1}
b={"b":1}
c={"c":1}


def A(a):
    return(a[0]+a[1]+a[2])

def B(a):
    return(a[0]-a[1]-a[2])

def C(a):
    return(a[0]*a[1]*a[2])

q=[A,B,C]

def H(nums,func_list):
    #print(func_list[2](nums))
    pass
    
H([1,2,3],q)

a=np.eye(3)*2+np.ones((3,3))
b=np.eye(3)*4+np.ones((3,3))
c=np.eye(3)*6+np.ones((3,3))
d=np.array([a,b,c])
e=np.eye(3)
row_stacks=[]
for r in range(3):
    row_matrices=[]
    for c in range(3):
        row_matrices.append(d[c]*e[r,c])
    row_stacks.append(np.concatenate(row_matrices,1))
f=np.concatenate(row_stacks,0)
#print(f)

def Propogate_Uncertainty(variables,constants,derivative_funcs,domain_length):
    key_list=set().union(*(d.keys() for d in variables))
    
    if("Jacobian" in key_list):
        derivative_matrices=[]
        covariance_matrices=[]
        
        for idx,variable in enumerate(variables):
            if(("Jacobian" in variable.keys())or("Uncertainty" in variable.keys())):
                derivative_matrices.append(np.diag(derivative_funcs[idx](variables,constants,domain_length)))
                
                if("Jacobian" in variable.keys()):
                    covariance_matrices.append(variable["Jacobian"])
                    
                else:
                    covariance_matrices.append(np.diag(np.power(variable["Uncertainty"],2)))
        
        derivative_matrix=np.concatenate(derivative_matrices,1)
        
        covariance_template=np.eye(len(covariance_matrices))
        row_stacks=[]
        for r in range(len(covariance_matrices)):
            row_matrices=[]
            for c in range(len(covariance_matrices)):
                row_matrices.append(covariance_matrices[c]*covariance_template[r,c])
            row_stacks.append(np.concatenate(row_matrices,1))
        covariance_matrix=np.concatenate(row_stacks,0)
        
        return({"Jacobian":derivative_matrix@covariance_matrix@(derivative_matrix.T)})
    
    elif("Uncertainty" in key_list):
        running_uncertainty=np.zeros(domain_length)
        for idx,variable in enumerate(variables):
            if("Uncertainty" in variable.keys()):
                running_uncertainty=running_uncertainty+(np.power(derivative_funcs[idx](variables,constants,domain_length),2)*np.power(variable["Uncertainty"],2))
        
        return({"Uncertainty":np.sqrt(running_uncertainty)})
    
    return({})

def a_der(inputs,constants,domain_length):
    return(np.ones(domain_length))

def b_der(inputs,constants,domain_length):
    return(np.ones(domain_length))

a={"Distribution":np.array([1,2,3]),
   "Uncertainty":np.array([0.5,0.5,0.5])}
b={"Distribution":np.array([2,4,6]),
   "Uncertainty":np.array([1,1,1])}
c=10

d={"Distribution":a["Distribution"]+b["Distribution"]+c}
print(d["Distribution"])

print(Propogate_Uncertainty([a,b],[c],[a_der,b_der],3))

a={"Distribution":np.array([1,2,3]),
   "Uncertainty":np.array([0.5,0.5,0.5])}
b={"Distribution":np.array([2,4,6]),
   "Jacobian":np.diag(np.array([1,1,1]))}
c=10

d={"Distribution":a["Distribution"]+b["Distribution"]+c}
print(d["Distribution"])

print(Propogate_Uncertainty([a,b],[c],[a_der,b_der],3))

a={"Distribution":np.array([1,2,3]),
   "Jacobian":np.diag(np.array([0.5,0.5,0.5]))}
b={"Distribution":np.array([2,4,6]),
   "Jacobian":np.diag(np.array([1,1,1]))}
c=10

d={"Distribution":a["Distribution"]+b["Distribution"]+c}
print(d["Distribution"])

print(Propogate_Uncertainty([a,b],[c],[a_der,b_der],3))