import pandas as pd

# Carregar o arquivo Excel
file_path = "Template - Requisito de Software v02.csv"
xls = pd.ExcelFile(file_path)

# Carregar abas específicas
sobre_documento = pd.read_excel(xls, sheet_name="Sobre o documento")
glossario = pd.read_excel(xls, sheet_name="1-Glossário")
requisitos = pd.read_excel(xls, sheet_name="2-Requisitos de Software")
lists = pd.read_excel(xls, sheet_name="3-Lists")
versoes = pd.read_excel(xls, sheet_name="4-Versões")
guia = pd.read_excel(xls, sheet_name="5-Guia")

# Exibir as primeiras linhas da aba de requisitos
print(requisitos.head())