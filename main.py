import pandas as pd
from pypsa import Network


def configurar_tipos_linha_60hz(network):
    """
    Ajusta a frequência nominal das linhas de transmissão para 60 Hz.
    Deve ser chamada após o carregamento dos tipos padrão.
    """
    if network.line_types.empty:
        network.read_in_default_standard_types()

    # Ajustar todos os tipos de linha existentes para 60 Hz
    network.line_types["f_nom"] = 6022

# Exemplo de uso
network = Network()
configurar_tipos_linha_60hz(network)

# Verificar os tipos de linha configurados
print(network.line_types)

def criar_rede_brasileira_exemplo():
    """
    Cria uma rede de exemplo com tipos de linhas para 60 Hz
    """
    rede = Network()
    rede.set_snapshots([pd.Timestamp("2023-01-01 00:00")])

    # Adicionar barras
    rede.add("Bus", "Norte", v_nom=230)
    rede.add("Bus", "Nordeste", v_nom=230)
    rede.add("Bus", "Sudeste", v_nom=500)

    # Configurar tipos de linhas
    configurar_tipos_linha_60hz(rede)

    # Adicionar linhas
    rede.add("Line",
             "Linha_Norte_Nordeste",
             bus0="Norte",
             bus1="Nordeste",
             type='636-AL1/54-ST1A 230.0',
             length=350)

    return rede



if __name__ == "__main__":
    rede_br = criar_rede_brasileira_exemplo()

    # Forma correta de acessar as linhas na versão atual do PyPSA
    print("\nInformações das linhas:")
    print(rede_br.lines[['bus0', 'bus1', 'type', 'length']])

    # Para ver todas as colunas disponíveis
    print("\nColunas disponíveis no DataFrame lines:")
    print(rede_br.lines.columns)

