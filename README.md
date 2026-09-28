The is an fea cad optimizer thats in a build123d part script, which scores the design variables of the part and a solid part. 

The GUI also contains editable variables like steps, and min max bounds. The optimizer also takes in a profile .json file that stores information about the simulation like the material its made out of, and its support and load faces. 

The FEA simulation returns stress and displacement values, which then get treated as constraints around the optimization of the mass ( trying to make it as small as possible while also keeping stress and displacement before their max's)

For optimization you can either use slsqp or gradient descent depending on your preference. 

<img width="2562" height="1736" alt="convergence" src="https://github.com/user-attachments/assets/83e493ef-d6fa-4ee8-bae5-c7e358c3c25f" />


Heres some of the math that powers it:
<img width="793" height="1024" alt="image" src="https://github.com/user-attachments/assets/af454753-be80-4a7d-9cd3-e3b1d6fbc2a2" />

<img width="793" height="1024" alt="image" src="https://github.com/user-attachments/assets/46ff5c10-05b2-456f-852a-896139f7f30c" />

I came up with the concept of this but apart from doing some of the math for the optimization stuff claude did most of the actual programming implementation. 

