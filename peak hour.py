import pypsa
import pandas as pd
import numpy as np

network = pypsa.Network()

# Lista simples com nomes dos subsistemas
subsistemas = ["N", "NE", "SE/CO", "S"]

# Adicionar barramentos
for subsis in subsistemas:
    network.add("Bus",
               name=f"Barramento_{subsis}",
               v_nom=500000)  # Tensão nominal de 500 kV
conexoes = [
    ('N', 'NE'),  # Norte conectado ao Nordeste
    ('N', 'SE/CO'),  # Norte conectado ao SE/CO
    ('NE', 'SE/CO'),  # Nordeste conectado ao SE/CO
    ('S', 'SE/CO')  # Sul conectado ao SE/CO
]
for origem, destino in conexoes:
    network.add("Line",
                name=f"LT_{origem}_{destino}",
                bus0=f"Barramento_{origem}",
                bus1=f"Barramento_{destino}",
                x=13*1.5,
                r=276*1.5,
                )
try:
    df_geracao = pd.read_excel("dados por subsistema 2.xlsx", index_col=0)

    # 4. Adicionar barramentos de geração para cada tipo em cada subsistema
    tipos_geracao = df_geracao.index.tolist()

    for subsis in subsistemas:
        for tipo in tipos_geracao:
            # Criar barramento específico para o tipo de geração
            bus_name = f"Geracao_{tipo}_{subsis}"
            network.add("Bus",
                        name=bus_name,
                        v_nom=500000,
                        )

            # Conectar ao barramento principal do subsistema
            network.add("Line",
                        name=f"Trafo_{bus_name}",
                        bus0=f"Geracao_{tipo}_{subsis}",
                        bus1=f"Barramento_{subsis}",
                        x=0.013,
                        r=0.276,
                        )

            # Adicionar gerador (se houver capacidade instalada)
            capacidade = df_geracao.loc[tipo, subsis]
            controle = "Slack" if (tipo == "UHE" and subsis == "SE/CO") else "PQ"
            if capacidade >= 0:
                network.add("Generator",
                            name=f"Ger_{tipo}_{subsis}",
                            bus=bus_name,
                            p_set=capacidade,
                            control=controle,
                            )

except FileNotFoundError:
    print("Arquivo 'dados por subsistema.xlsx' não encontrado. Criando rede apenas com barramentos principais.")

cargas = {
    'N': 7979.966,
    'NE': 1514.118,
    'S': 18348.577,
    'SE/CO': 57590.606
}

for subsis, carga in cargas.items():
    network.add("Load", f"load{subsis}",
                bus=f"Barramento_{subsis}",
                p_set=carga*0.92,
                q_set=carga*0.39)
network.pf()

# Coleta resultados em DataFrames
v_mag = network.buses_t.v_mag_pu
v_ang = network.buses_t.v_ang
p0 = network.lines_t.p0
q0 = network.lines_t.q0
p_gen = network.generators_t.p
p_load = network.loads_t.p

# Cria um arquivo Excel com várias abas
with pd.ExcelWriter("fluxo_de_potencia 19.xlsx") as writer:
    v_mag.to_excel(writer, sheet_name="Tensão Magnitude")
    v_ang.to_excel(writer, sheet_name="Tensão Ângulo")
    p0.to_excel(writer, sheet_name="Fluxo Ativo Linhas")
    q0.to_excel(writer, sheet_name="Fluxo Reativo Linhas")
    p_gen.to_excel(writer, sheet_name="Potência Geradores")
    p_load.to_excel(writer, sheet_name="Potência Cargas")