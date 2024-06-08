import streamlit as st
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem, EState
import pandas as pd
import numpy as np
import pickle

# Load the pre-trained model
model_path = 'D:/peptide/Web App/trained_model.sav'
with open(model_path, 'rb') as model_file:
    model = pickle.load(model_file)

# Define a function to calculate all molecular descriptors
def calculate_all_descriptors(mol):
    descriptor_names = [desc_name for desc_name, _ in Descriptors.descList]
    descriptors = {}
    for desc_name, desc_func in Descriptors.descList:
        try:
            descriptors[desc_name] = desc_func(mol)
        except Exception as e:
            descriptors[desc_name] = str(e)
    return descriptors

# Add a sidebar with navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Description", "Database", "Contact"])

# Home page
if page == "Home":
    st.title("Predicting Permeability of Cyclic Peptides")
    st.subheader("Calculating molecular descriptors from SMILES string")

    # User input: SMILES string
    smiles_input = st.text_input("Enter SMILES string:", "")

    if smiles_input:
        try:
            # Convert SMILES string to RDKit molecule object
            mol = Chem.MolFromSmiles(smiles_input)

            if mol:
                # Calculate all descriptors
                descriptors = calculate_all_descriptors(mol)

                # Convert descriptors to DataFrame
                df = pd.DataFrame(descriptors.items(), columns=["Descriptor", "Value"])

                # # Display descriptors
                # st.write("### Molecular Descriptors")
                # st.dataframe(df)

                # Prepare the input for the model using selected descriptors
                model_input = np.array([[
                    descriptors['MinPartialCharge'],
                    descriptors['PEOE_VSA6'],
                    descriptors['SlogP_VSA8'],
                    descriptors['VSA_EState6'],
                    descriptors['VSA_EState9'],
                    descriptors['NumAromaticRings'],
                    descriptors['NumRotatableBonds'],
                    descriptors['MolLogP'],
                    descriptors['fr_Al_OH']
                ]])

                # Make prediction
                prediction = model.predict(model_input)
                permeability = "low" if prediction[0] == 0 else "good"
                st.write("### Prediction")
                st.write(f"The predicted permeability is: {permeability}")

                # Display descriptors
                st.write("### Molecular Descriptors")
                st.dataframe(df)
            else:
                st.error("Invalid SMILES string. Please enter a valid SMILES.")



        except Exception as e:
            st.error(f"An error occurred: {e}")

# Description page
elif page == "Description":
    st.title("Description")
    st.write("### Model Details")
    st.write("This section provides details about the machine learning model used.")

    # Model details
    model_details = {
        "Descriptors": ["MinPartialCharge", "PEOE_VSA6", "SlogP_VSA8", "VSA_EState6", "VSA_EState9", 
                        "NumAromaticRings", "NumRotatableBonds", "MolLogP", "fr_Al_OH"],
    }
    model_df = pd.DataFrame(model_details)
    st.write("#### Descriptors Used in the Model")
    st.table(model_df)

    # Model performance metrics
    performance_metrics = {
        "Metric": ["Accuracy", "F1 Score"],
        "Score": [0.81, 0.87]
    }
    performance_df = pd.DataFrame(performance_metrics)
    st.write("#### Model Performance")
    st.table(performance_df)

    # Permeability criteria
    permeability_criteria = {
        "Permeability": ["Good", "Low"],
        "Criteria": [">=-6", "<-6"]
    }
    permeability_df = pd.DataFrame(permeability_criteria)
    st.write("#### Permeability Criteria")
    st.table(permeability_df)

# Database page
elif page == "Database":
    st.title("Database")
    st.write("This section includes information about the molecular database.")
    st.write("CycPeptMPDB (Cyclic Peptide Membrane Permeability Database) is the largest web-accessible database of membrane permeability of cyclic peptide. The latest version provides the information for 7,334 structurally diverse cyclic peptides collected from 47 publications. These cyclic peptides are composed of 312 types Monomers (substructures). ")
    st.write("### Source of Training Dataset")
    st.write("[Cyclic Peptide Database](http://cycpeptmpdb.com/)")

    df = pd.read_csv("CycPeptMPDB_Peptide.csv")
    st.write(df.head())

    # st.write("### Data summary")
    
    # # Placeholder for three images and their captions
    # st.image("D:\peptide\code\err_02\Countplot_permeability.jpg", width = 400, caption="~2000 peptides are having low permeability and 4500 peptides are having high permeability")
    # st.image("D:\peptide\code\err_02\DistPlot.jpg", caption="Distribution of permeability values from the database")
    # st.image("D:\peptide\code\err_02\Scatter_MOlLoGP.jpg", width = 400, caption="Correlation between the MolLog P and permeability values")


# Contact page
elif page == "Contact":
    st.title("Contact")
    st.write("You can reach me at:")
    st.write("Email: Hj728490@gmail.com")
    st.write("Mobile: +91 9024990040")
    st.write("We are open to any Feedbacks, suggestions and contributions",style = "center")