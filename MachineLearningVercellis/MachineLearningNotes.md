# Machine Learning  - Carlo Vercellis

- **Con riferimento alla regressione Ridge, quali affermazioni sono corrette? **

  1. (False) Richiede la minimizzazione della funzione  
     $$
     min_{w} \lambda||w||+(y-Xw)'(y-Xw)
     $$

  2. (True) Risulta utile quando l'inversione della matrice **X'X** costituisce un problema mal condizionato.

  3. (True) Richiede la minimizzazione della funzione
     $$
     min_{w} \lambda||w||^{2}+(y-Xw)'(y-Xw)
     $$

  4. (False) Non si può utilizzare se la matrice **X'X** è singolare

  5. (False) Richiede la minimizzazione della funzione 
     $$
     min_{w} ||w||+(y-Xw)'(y-Xw)
     $$

  6. (False) Nessuna risposta è giusta.

  * *Ricorda che nella regressione Ridge:*
    $$
    \hat{\boldsymbol{w}} = (\boldsymbol{X}^T\boldsymbol{X}+\lambda\boldsymbol{I})^{-1}\boldsymbol{X}^T\boldsymbol{y}.
    $$

  * Invece nella lr normale...

  $$
  \hat{\boldsymbol{w}} = (\boldsymbol{X}^T\boldsymbol{X})^{-1}\boldsymbol{X}^T\boldsymbol{y}.
  $$

  

- **Quale tra le seguenti affermazioni è vera?**

  1. Il coefficiente di correlazione lineare tra due attributi è definito come 
     $$
     r_{jk} = corr(\textbf{a}_{j},\textbf{a}_{k}) = \frac{v_{jk}}{\sigma_{j}\sigma_{k}}
     $$
     