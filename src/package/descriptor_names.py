

import streamlit as st
import pandas as pd

import sys
sys.path.append('/home/ganesh')

import maha_power
from maha_power.Descriptors import Descriptor_calculation
from maha_power.Fingureprint import Fingureprint_calculation
from maha_power.Descriptor_set import Descriptor_set_calculation
from maha_power.Pyscf_Based_QM_parameters import Pyscf_Based_QM_parameters_calculation

descriptor_methods = {
    "NitrogenRuleCalculator1": Descriptor_calculation.NitrogenRuleCalculator1,
    "MolecularDescriptors": Descriptor_calculation.MolecularDescriptors,
    "DIPOL_RI": Descriptor_calculation.DIPOL_RI,
    "M_Power_RDkit_Mol_Property": Descriptor_calculation.M_Power_RDkit_Mol_Property,
    "Mol_Descriptor": Descriptor_calculation.Mol_Descriptor,
    "Halogen_Descriptors": Descriptor_calculation.Halogen_Descriptors,
    "Filling_Atomic_Orbitals": Descriptor_calculation.Filling_Atomic_Orbitals,
    "Hypothetical_Molecular_Activity": Descriptor_calculation.Hypothetical_Molecular_Activity,
    "Molar_Activity_Surface_Area": Descriptor_calculation.Molar_Activity_Surface_Area,
    "Charge_Distribution_Descriptor": Descriptor_calculation.Charge_Distribution_Descriptor,
    "Electronic_Transition_and_Ionization": Descriptor_calculation.Electronic_Transition_and_Ionization,
    "LewisAcidBaseAnalyzer": Descriptor_calculation.LewisAcidBaseAnalyzer,
    "Polarity_Energy_Interaction": Descriptor_calculation.Polarity_Energy_Interaction,
    "MolecularPropertiesCalculator": Descriptor_calculation.MolecularPropertiesCalculator,
    "Electron_orbital_Zeff": Descriptor_calculation.Electron_orbital_Zeff,
    "BFGS_Dielectric_Constants": Descriptor_calculation.BFGS_Dielectric_Constants
}

fingerprint_methods = {
    "FINGERPRINT": Fingureprint_calculation.FINGERPRINT,
    "FingerprintCalculator": Fingureprint_calculation.FingerprintCalculator,
    "PubChemFingerprintCalculator": Fingureprint_calculation.PubChemFingerprintCalculator,
    "sombor_indices": Fingureprint_calculation.sombor_indices
}

qm_methods = {
    "HOMO_LUMO_Energy": Pyscf_Based_QM_parameters_calculation.HOMO_LUMO_Energy,
    "ElectronicPropertiesCalculator": Pyscf_Based_QM_parameters_calculation.ElectronicPropertiesCalculator
}

descriptor_set_methods = {
    "DescriptorCalculator": Descriptor_set_calculation.DescriptorCalculator,
    "MolecularDescriptors": Descriptor_set_calculation.MolecularDescriptors
}

# # Create the Streamlit sidebar for selections
# st.sidebar.title("Select Calculation Methods")

# selected_descriptor_methods = st.sidebar.multiselect(
#     "Select Descriptor Methods",
#     options=list(descriptor_methods.keys())
# )

# selected_fingerprint_methods = st.sidebar.multiselect(
#     "Select Fingerprint Methods",
#     options=list(fingerprint_methods.keys())
# )

# # Additional options for FingerprintCalculator, placed immediately after the fingerprint method selection
# if "FingerprintCalculator" in selected_fingerprint_methods:
#     with st.sidebar.container():
#         fingerprint_options = ['ecfp0', 'ecfp2', 'ecfp4', 'ecfp6', 'ecfc0', 'ecfc2', 'ecfc4', 'ecfc6', 
#                                'fcfp2', 'fcfp4', 'fcfp6', 'fcfc2', 'fcfc4', 'fcfc6', 'lecfp4', 'lecfp6', 
#                                'lfcfp4', 'lfcfp6', 'maccs', 'ap', 'tt', 'hashap', 'hashtt', 'avalon', 
#                                'laval', 'rdk5', 'rdk6', 'rdk7']
#         selected_fingerprint_types = st.sidebar.multiselect(
#             "Select Fingerprint Types",
#             options=fingerprint_options
#         )

# selected_qm_methods = st.sidebar.multiselect(
#     "Select QM Methods",
#     options=list(qm_methods.keys())
# )

# selected_descriptor_set_methods = st.sidebar.multiselect(
#     "Select Descriptor Set Methods",
#     options=list(descriptor_set_methods.keys())
# )

# # Upload data
# uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

# if uploaded_file is not None:
#     df = pd.read_csv(uploaded_file)
#     st.write("Uploaded DataFrame:")
#     st.write(df)

#     if st.button("Run Selected Calculations"):
#         for method_name in selected_descriptor_methods:
#             st.write(f"Running {method_name}...")
#             df = descriptor_methods[method_name](df)

#         for method_name in selected_fingerprint_methods:
#             st.write(f"Running {method_name}...")
#             if method_name == "FingerprintCalculator":
#                 if selected_fingerprint_types:
#                     df = fingerprint_methods[method_name](df, selected_fingerprint_types)
#                 else:
#                     st.warning("Please select at least one fingerprint type for FingerprintCalculator.")
#             else:
#                 df = fingerprint_methods[method_name](df)

#         for method_name in selected_qm_methods:
#             st.write(f"Running {method_name}...")
#             df = qm_methods[method_name](df)

#         for method_name in selected_descriptor_set_methods:
#             st.write(f"Running {method_name}...")
#             df = descriptor_set_methods[method_name](df)

#         st.write("Resulting DataFrame:")
#         st.write(df)
        
#         # Display the shape of the resulting DataFrame
#         st.markdown(f"**DataFrame Shape:** {df.shape}")
