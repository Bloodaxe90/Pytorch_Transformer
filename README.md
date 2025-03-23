Pytorch Transformer:

To test my Transformer knowledge I build a transformer model with pytorch (no hugging face),
this inludes dot product attention, positional encoding and layer normalization.

The final model ended up being a decoder based transformer trained to generate characters for Shakespears Macbeth,
for fun I also trained model on the Bee Movies script and Hamilton Script. 

Results can be found in the "inference" jupyter notebook in the notebooks directory for all 3 models.

I will add that the dot product attention class has options for scaled and masked attention aswell as cross attention
however this was never tested so I am unsure if it actually works.

