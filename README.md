# Support Ticket Classification & Priority Prediction

## Overview
This project builds a machine learning system to automatically classify customer 
support tickets into categories (Billing, Technical, Refund, etc.) and predict 
their priority level (Low/Medium/High/Critical) using NLP techniques.

Built as part of Future Interns ML Task 2 (2026).

## Dataset
[Customer Support Ticket Dataset (Kaggle)](https://www.kaggle.com/datasets/suraj520/customer-support-ticket-dataset)
- 8,469 support tickets
- Fields used: Ticket Subject, Ticket Description, Ticket Type, Ticket Priority

## Approach
1. **Preprocessing**: Combined subject + description, lowercased text, removed 
   punctuation/numbers, removed stopwords (including custom templated boilerplate 
   tokens found in the raw data, e.g. "productpurchased").
2. **Feature Extraction**: TF-IDF vectorization (top 3,000 features).
3. **Modeling**: Logistic Regression, trained separately for:
   - Ticket category (`Ticket Type`) — 5 classes
   - Priority level (`Ticket Priority`) — 4 classes
4. **Evaluation**: Train/test split (80/20, stratified), classification report, 
   confusion matrix.

## Results
**Target           Accuracy  Random Baseline **

Ticket Type      ~20%      ~20% (5 classes) |
Ticket Priority  ~24%      ~25% (4 classes) |

Both models perform at approximately chance level. The confusion matrix (see 
`/notebook`) shows errors spread uniformly across all classes, with no diagonal 
concentration — indicating no learnable relationship between ticket text and 
either label.

## Key Finding
This dataset's ticket text appears synthetically generated and not causally 
linked to its assigned category or priority labels. This is a known limitation 
of this particular Kaggle dataset. A dataset built from real, human-written 
tickets and human-assigned labels (e.g. the Zenodo IT Support Ticket dataset) 
would likely show genuine text-to-label correlation.

## Tools Used
- Python, Jupyter Notebook
- pandas, NLTK, scikit-learn (TF-IDF, Logistic Regression)
- matplotlib (confusion matrix visualization)

## How to Run
1. Clone this repo
2. Download the dataset from the Kaggle link above and place it in `/data`
3. Open `ticket_classification.ipynb` and run all cells
