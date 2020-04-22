# Community Learning
> Ziel ist es mit diesem kleine PoC aufzuzeigen wie mit Hilfe von Federated Learning bessere Prognose erzielt werden können. 


Hier ein Auszug von der Kaggle Website
> In this competition, you are provided with 1.5 years of customers behavior data from Santander bank to predict what new products customers will purchase. The data starts at 2015-01-28 and has monthly records of products a customer has, such as "credit card", "savings account", etc. You will predict what additional products a customer will get in the last month, 2016-06-28, in addition to what they already have at 2016-05-28. These products are the columns named: ind_(xyz)_ult1, which are the columns #25 - #48 in the training data. You will predict what a customer will buy in addition to what they already had at 2016-05-28. 

## Machine Learning Problem
Wie weiter oben beschrieben ist das Ziel die zusätzlich gekauften Produkte im Monat vom 2016-06-28 zu bestimmen. Die Daten beinhalten Montasdaten von 2015-01-28 - 2016-05-28 (ca. 1.5 Jahre). Nun gäbe es mehrere Möglichkeiten das ML Problem zu formulieren: 
- wir versuchen immer die gekauften Produkte des nächsten Monats zu bestimmen
- wir versuchen immer die gekauften Produkte für den Juni jeweils anhand des Vormonats zu bestimmen (ignorieren alle anderen Monate). 
- wir trainieren ein recommender system (collaborative filtering) mit allen Daten der Produkte und versuchen den letzten Monat vorherzusagen.

## Installation

## Vorgehen PoC

Um den Usecase möglichst realistisch zu gestalten, gehen wir wie folgt vor:


**Variante 1:**
1. **Daten bereitstellen und bereinigen**: Hierzu werden wir das Datenset so aufteilen, dass je ein Datenset pro Bank entsteht. Dazu werden wir ein geografisches Attribut hernehmen. Danach werden die Daten nochmals im Verhältnis 80/20 aufgeteilt in ein Train- und Testset (`data_bank1_train`, `data_bank1_test`, `data_bank2_train`, `data_bank2_test`). 
2. **Baseline Modelle trainiern:** Pro Bank werden wir einen GradientBoost Algorithmus trainieren mit deren Default-Einstellungen. Dadurch erhaltne wir 2 Modelle (`model_bank1` und `model_bank2`)
3. **Ensemble Predictions:** In diesem Schritt werden wir die Resultate von model_bank1 und model_bank2 kombinieren. 
 - `model_bank1` und `model_bank2` wird mit den `data_bank1_test` gefüttert und eine gemeinsame Prediction erstellt.
 - `model_bank1` und `model_bank2` wird mit den `data_bank2_test` gefüttert und eine gemeinsame Prediction erstellt.
4. **Auswertung:**: Um festzustellen ob das Ensemble eine Mehrwert bringt werden folgende Resultate verglichen.
 - `model_bank1(data_bank1_test)` vs `ensemble(model_bank1(data_bank1_test), model_bank2(data_bank1_test)`
 - `model_bank2(data_bank2_test)` vs `ensemble(mdoel_bank2(data_bank1_test), model_bank2(data_bank2_test)`

## How to use
