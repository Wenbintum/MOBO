## Dataset structure

Pytorch_geometric and pickle are needed to load the dataset

```python
Data(

    x=[56, 14],           #14-dimensional node features

    edge_index=[2, 414],  #edges of graph representations 
    
    y=[1],                #adsorption enthalpy
    
    site='fcc',           #fcc hollow O* site
    
    ads='O',              #adsorbate is O*
    
    ens={                 #three Pd atoms are active site atoms.
      Ag=0,
      Au=0,
      Cu=0,
      Ir=0,
      Os=0,
      Pd=3,
      Pt=0,
      Re=0,
      Rh=0,
      Ru=0
    }
    )
```
