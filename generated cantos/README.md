Understanding generated text files
==================================

This folder contains all the generated text we collect from our models, grouped by subfolder named as 
model params <-- {num_layers_encoder}_{num_layers_decoder}_{d_model}_{dff}_{num_heads} _ {repetitions_production} _ {repetitions_comedy}

In each subfolder we stored 11 different generations, at different temperatures in the range (0.5, 1.5), each one in a .txt file, named as
canto_temp_{temperature}_[<model params>].txt

Plus, we provide a .json file containing the details of the training of the model, such as model params, training epochs (and repetitions) 
on productions and on the Divine Comedy, loss and accuracy history, and the encapsulated generated cantos, for each temperature. 

This has be done in order to both making accessible the training info and encapsulating all the generated text into the same file. 
The canto_xxx.txt files are useful only for an individual inspections since are already formatteed and easy to be read. 
