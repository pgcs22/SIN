import pypsa
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point, LineString, MultiLineString
import re
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pypsa.components import Network
import logging

def plotar_mapa_sin(linhas, subestacoes, geradores, mostrar_geradores=True, titulo="Sistema Interligado Nacional"):
    """
    Plota o mapa do Brasil com os ativos do SIN.

    Args:
        linhas: GeoDataFrame com as linhas de transmissão
        subestacoes: GeoDataFrame com as subestações
        geradores: GeoDataFrame com os geradores
        mostrar_geradores: Bool para mostrar ou não os geradores no mapa
        titulo: Título do gráfico
    """
    try:
        # Configurações iniciais
        plt.figure(figsize=(15, 15))
        ax = plt.gca()

        # Carregar shapefile do Brasil (limites estaduais)
        try:
            brasil = gpd.read_file(r"C:\Users\pgcs_\PycharmProjects\PyPSA\raw\BR_UF_2022.shp")

            brasil.plot(ax=ax, color='lightgray', edgecolor='black', alpha=0.5)
        except:
            # Fallback caso não consiga carregar o shapefile
            pass

        # Plotar linhas de transmissão
        linhas.plot(ax=ax,
                    column='tipo',
                    legend=True,
                    linewidth=0.5,
                    cmap='viridis',
                    legend_kwds={'title': 'Tipo de Linha', 'loc': 'lower right'})

        # Plotar subestações
        subestacoes.plot(ax=ax,
                         column='tipo',
                         markersize=20,
                         legend=True,
                         cmap='Set1',
                         legend_kwds={'title': 'Tipo de Subestação', 'loc': 'lower left'})

        # Plotar geradores (opcional)
        if mostrar_geradores:
            geradores.plot(ax=ax,
                           color='red',
                           markersize=5,
                           alpha=0.7,
                           label='Geradores')

            # Adicionar legenda manual para geradores
            legend_elements = [Line2D([0], [0],
                                      marker='o',
                                      color='w',
                                      label='Geradores',
                                      markerfacecolor='red',
                                      markersize=10)]
            ax.legend(handles=legend_elements, loc='upper right')

        # Configurações do gráfico
        plt.title(titulo, fontsize=16)
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.grid(True, alpha=0.3)

        # Melhorar os limites do mapa
        xmin, ymin, xmax, ymax = linhas.total_bounds
        plt.xlim(xmin - 1, xmax + 1)
        plt.ylim(ymin - 1, ymax + 1)

        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Erro ao plotar mapa: {e}")
        raise

def plotar_mapa_detalhado(linhas, subestacoes, geradores):
    """
    Versão corrigida com conversão de CRS para todos os elementos
    """
    import matplotlib.pyplot as plt

    # 1. Configuração inicial
    plt.figure(figsize=(20, 20))
    ax = plt.gca()

    try:
        # 2. Definir CRS alvo (UTM 23S para Brasil)
        CRS_ALVO = "EPSG:31983"

        # 3. Carregar e preparar Brasil
        brasil = gpd.read_file(r"C:\Users\pgcs_\PycharmProjects\PyPSA\raw\BR_UF_2021.shp")
        #brasil = brasil.to_crs(CRS_ALVO)  # Converter para UTM

        # 4. Converter todos os ativos para o mesmo CRS
        linhas = linhas.to_crs(CRS_ALVO)
        subestacoes = subestacoes.to_crs(CRS_ALVO)
        geradores = geradores.to_crs(CRS_ALVO)

        # 5. Verificação de dados
        print("\nVerificação de CRS:")
        print(f"Brasil: {brasil.crs}")
        print(f"Linhas: {linhas.crs}")
        print(f"Subestações: {subestacoes.crs}")
        print(f"Geradores: {geradores.crs}")

        # 6. Plotagem garantida
        brasil.plot(ax=ax, color='#f0f0f0', edgecolor='black', linewidth=0.5)

        # Linhas (coordenadas agora compatíveis)
        linhas.plot(
            ax=ax,
            color='blue',
            linewidth=0.5,
            label='Linhas'
        )

        # Subestações
        subestacoes.plot(
            ax=ax,
            color='red',
            markersize=10,
            marker='s',
            label='Subestações'
        )

        # Geradores
        geradores.plot(
            ax=ax,
            color='yellow',
            markersize=25,
            marker='o',
            edgecolor='black',
            label='Geradores'
        )

        # 7. Ajuste de limites com base no Brasil
        xmin, ymin, xmax, ymax = brasil.total_bounds
        margem = 200000  # 200 km de margem
        ax.set_xlim(xmin - margem, xmax + margem)
        ax.set_ylim(ymin - margem, ymax + margem)

        # 8. Configurações visuais
        plt.title('Sistema Interligado Nacional - Escala Corrigida', fontsize=18)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Erro durante a plotagem: {str(e)}")
# o objetivo desse código é modelar o SIN no PyPSA. Primeiro passo é carregar os ativos

def carregar_LTs():
    try:
        # Linhas de transmissão
        linhas_atuais = gpd.read_file(
            r"C:\Users\pgcs_\PycharmProjects\PyPSA\raw\Linhas_de_Transmissão___Base_Existente.shp"
        )
        linhas_futuras = gpd.read_file(
            r"C:\Users\pgcs_\PycharmProjects\PyPSA\raw\Linhas_de_Transmissão___Expansão_Planejada.shp"
        )

        # Adiciona uma coluna para identificar se a linha é atual ou futura (opcional)
        linhas_atuais['tipo'] = 'existente'
        linhas_futuras['tipo'] = 'planejada'

        # Junta os dois GeoDataFrames
        linhas_completas = pd.concat([linhas_atuais, linhas_futuras], ignore_index=True)
        linhas_completas = linhas_completas.drop(['Ano_Opera', 'created_da', 'last_edite', 'last_edi_1'], axis=1)

        linhas_completas.to_excel('linhas.xlsx')

        return linhas_completas

    except Exception as e:
        print(f"Erro ao carregar arquivos shapefile: {e}")
        exit()

def extrair_pontos_referencia(gdf_linhas):
    """
    Extrai pontos de origem e destino de geometrias LineString/MultiLineString.

    Args:
        gdf_linhas (GeoDataFrame): GeoDataFrame contendo linhas de transmissão

    Returns:
        GeoDataFrame: Novo GeoDataFrame com colunas 'origem_geom' e 'destino_geom' contendo pontos
    """

    # Verifica se o GeoDataFrame tem geometria
    if not isinstance(gdf_linhas, gpd.GeoDataFrame):
        raise TypeError("A entrada deve ser um GeoDataFrame")

    if 'geometry' not in gdf_linhas.columns:
        raise ValueError("O GeoDataFrame deve ter uma coluna 'geometry'")

    # Cria cópia para não modificar o original
    gdf = gdf_linhas.copy()

    # Cria colunas para os pontos
    gdf['origem_geom'] = None
    gdf['destino_geom'] = None

    for idx, row in gdf.iterrows():
        geom = row.geometry

        # Caso LineString simples
        if isinstance(geom, LineString):
            coords = list(geom.coords)
            if len(coords) >= 2:
                gdf.at[idx, 'origem_geom'] = Point(coords[0])
                gdf.at[idx, 'destino_geom'] = Point(coords[-1])

        # Caso MultiLineString
        elif isinstance(geom, MultiLineString):
            # Pega a primeira coordenada da primeira linha e última da última linha
            first_line = geom.geoms[0]
            last_line = geom.geoms[-1]

            first_coords = list(first_line.coords)
            last_coords = list(last_line.coords)

            if first_coords and last_coords:
                gdf.at[idx, 'origem_geom'] = Point(first_coords[0])
                gdf.at[idx, 'destino_geom'] = Point(last_coords[-1])

        # Caso geometria inválida
        else:
            print(f"Aviso: Geometria não suportada na linha {idx} - {type(geom)}")

    # Converte para GeoDataFrame as novas colunas de geometria
    gdf.set_geometry('origem_geom', inplace=True)
    gdf['destino_geom'] = gpd.GeoSeries(gdf['destino_geom'])

    return gdf

def carregar_SEs():
    try:
        # Subestações
        SEs_atuais = gpd.read_file(
            r"C:\Users\pgcs_\PycharmProjects\PyPSA\raw\Subestações___Base_Existente.shp"
        )
        SEs_futuras = gpd.read_file(
            r"C:\Users\pgcs_\PycharmProjects\PyPSA\raw\Subestações___Expansão_Planejada.shp"
        )

        # Adiciona uma coluna para identificar se a linha é atual ou futura (opcional)
        SEs_atuais['tipo'] = 'existente'
        SEs_futuras['tipo'] = 'planejada'

        # Junta os dois GeoDataFrames
        SEs_completas = pd.concat([SEs_atuais, SEs_futuras], ignore_index=True)
        SEs_completas = SEs_completas.drop(['Ano_Opera'], axis=1)

        def classificar_subsistema(lon, lat):
            """Classificação melhorada por coordenadas geográficas"""
            # Região Nordeste (mais restrita)
            if (-46 < lon < -34) and (-18 < lat < -1):
                return 'NE'

            # Região Norte (área ampla)
            elif (lon < -50) and (-12 < lat < 5) or \
                    (-60 < lon < -50) and (-18 < lat < -12):
                return 'N'

            # Região Sudeste (incluindo Centro-Oeste)
            elif (-53 < lon < -39) and (-24 < lat < -12):
                return 'SE'

            # Região Sul
            elif (-57 < lon < -47) and (-34 < lat < -22):
                return 'S'

            # Default seguro (ajustar conforme necessidade)
            return 'SE'  # Considerando que a maioria das SEs estão no SE

        SEs_completas['Subsistema'] = SEs_completas.apply(
            lambda row: classificar_subsistema(row.geometry.x, row.geometry.y),
            axis=1
        )

        SEs_completas.to_excel('SEs.xlsx')

        return SEs_completas

    except Exception as e:
        print(f"Erro ao carregar arquivos shapefile: {e}")
        exit()

def carregar_Geracao():
    try:
        geradores = []
        operacao = ('Base_Existente', 'Expansão_Planejada')
        fontes = ('PCH', 'CGH', 'EOL', 'UFV', 'UHE', 'UTE_Biomassa', 'UTE_Fóssil', 'UTE_Nuclear')

        # Dicionário de mapeamento para padronizar os nomes das colunas
        colunas_padrao = {
            'NOME': 'nome',
            'nome': 'nome',
            'Nome': 'nome',
            'POTENCIA': 'potencia',
            'Potencia': 'potencia',
            'Potência': 'potencia',
            'potencia': 'potencia',
            'INI_OPER': 'ini_oper',
            'Ini_Oper': 'ini_oper',
            'ini_oper': 'ini_oper',
            'start_time': 'ini_oper'  # caso exista essa coluna em alguns arquivos
        }

        for op in operacao:
            for fonte in fontes:
                file_path = f'C:\\Users\\pgcs_\\PycharmProjects\\PyPSA\\raw\\{fonte}___{op}.shp'
                gdf = gpd.read_file(file_path)

                # Padroniza os nomes das colunas
                gdf.columns = [colunas_padrao.get(col, col) for col in gdf.columns]

                # Adiciona coluna com a fonte original (opcional)
                if fonte == 'EOL':
                    gdf['fonte_original'] = 'UEE'
                elif 'UTE' in str(fonte).upper():
                    gdf['fonte_original'] = 'UTE'
                else:
                    gdf['fonte_original'] = fonte

                if op == 'Base_Existente':
                    gdf['ini_oper'] = gdf['ini_oper'].replace(['-', 0.0], np.nan)

                # Colunas para remover (usando nomes padronizados)
                cols_to_drop = ['Shape_STLe', 'created_da', 'created_us', 'last_edite', 'last_edi_1', 'combust',
                                'COMBUST', 'Rio', ' Leilao', 'ano_prev', 'CEG', 'Leilão', 'leilao']
                gdf = gdf.drop(columns=[col for col in cols_to_drop if col in gdf.columns], errors='ignore')

                geradores.append(gdf)

        # Concatena todos os DataFrames
        gdf_final = pd.concat(geradores, axis=0).reset_index(drop=True)

        # Converte a potência para MW
        gdf_final['potencia'] = gdf_final['potencia'] / 1e3

        # Filtra apenas geradores com potência >= 50 MW (APÓS a conversão)
        gdf_final = gdf_final[gdf_final['potencia'] >= 10]
        gdf_final.to_excel('geração.xlsx')
        return gdf_final

    except Exception as e:
        print(f"Erro ao carregar arquivos shapefile: {e}")
        raise

def add_substations_to_pypsa(network, gdf_substations):
    """
    Adiciona subestações de um GeoDataFrame a uma rede PyPSA como barramentos,
    conectando diferentes níveis de tensão com transformadores e distribui cargas homogeneamente.

    Parâmetros:
    ----------
    network : pypsa.Network
        Rede PyPSA à qual as subestações serão adicionadas.
    gdf_substations : geopandas.GeoDataFrame
        GeoDataFrame contendo as subestações com suas informações, incluindo tensões e subsistema.

    Retorna:
    -------
    pypsa.Network
        Rede PyPSA modificada com as subestações e cargas adicionadas.
    """


    cargas_subsistema = {
        'N': 7605.14299999,
        'NE': 13158.44099999,
        'S': 14438.32500000,
        'SE': 45205.71099999
    }

    # Verificar se o GeoDataFrame tem as colunas necessárias
    required_columns = {'Nome', 'Tensao', 'Subsistema'}
    if not required_columns.issubset(gdf_substations.columns):
        raise ValueError(f"GeoDataFrame deve conter as colunas: {required_columns}")

    # Pré-processamento: Contar barramentos por subsistema
    subsistemas = gdf_substations['Subsistema'].unique()
    barramentos_por_subsistema = {sub: 0 for sub in subsistemas}

    # Primeira passada: contar quantos barramentos cada subsistema terá
    for _, substation in gdf_substations.iterrows():
        voltages = substation['Tensao']
        subsistema = substation['Subsistema']

        if isinstance(voltages, str):
            voltage_list = [float(v.strip().replace(',', '.')) for v in voltages.split('/')]
        elif isinstance(voltages, (list, np.ndarray)):
            voltage_list = [float(str(v).replace(',', '.')) for v in voltages]
        else:
            raise ValueError(f"Formato de tensão não suportado para subestação {substation['name']}")

        barramentos_por_subsistema[subsistema] += len(voltage_list)

    # Calcular carga por barramento em cada subsistema
    carga_por_barramento = {}
    for subsistema, carga_total in cargas_subsistema.items():
        if subsistema in barramentos_por_subsistema and barramentos_por_subsistema[subsistema] > 0:
            carga_por_barramento[subsistema] = carga_total / barramentos_por_subsistema[subsistema]
        else:
            carga_por_barramento[subsistema] = 0

    # Segunda passada: adicionar barramentos, transformadores e cargas
    for _, substation in gdf_substations.iterrows():
        name = substation['Nome']
        voltages = substation['Tensao']
        subsistema = substation['Subsistema']

        # Processar tensões
        if isinstance(voltages, str):
            voltage_list = [float(v.strip().replace(',', '.')) for v in voltages.split('/')]
            voltage_list = sorted(voltage_list, reverse=True)
        elif isinstance(voltages, (list, np.ndarray)):
            voltage_list = sorted([float(str(v).replace(',', '.')) for v in voltages], reverse=True)
        else:
            raise ValueError(f"Formato de tensão não suportado para subestação {name}")

        # Adicionar barramento para cada nível de tensão
        bus_names = []
        for i, voltage in enumerate(voltage_list):
            bus_name = f"{name} C{i + 1}"
            bus_names.append(bus_name)

            if bus_name not in network.buses.index:
                network.add("Bus",
                            name=bus_name,
                            v_nom=voltage,
                            v_mag_pu_set=0.98,
                            geometry=substation.geometry,
                            substation=name,
                            subsistema=subsistema)

                load_name = f"load_{bus_name}"
                network.add("Load",
                            name=load_name,
                            bus=bus_name,
                            p_set=carga_por_barramento[subsistema],
                            q_set=0)  # Considerando fator de potência 1

        # Adicionar transformadores entre barramentos de diferentes tensões
        for i in range(len(bus_names) - 1):
            bus0 = bus_names[i]
            bus1 = bus_names[i + 1]

            trafo_name = f"traf-{bus0}-{bus1}"

            if trafo_name not in network.transformers.index:
                network.add("Transformer",
                            name=trafo_name,
                            bus0=bus0,
                            bus1=bus1,
                            model="t",
                            r=0.005,
                            x=0.1,
                            b=0.05,
                            g=0)
    network.loads.to_excel('cargas_inseridas.xlsx')
    return network

def processar_nome_lt(nome_lt):
    """
    Processa o nome de uma Linha de Transmissão (LT) ou Ramal.
    Versão melhorada que trata múltiplos destinos (ramais).

    Retorna:
    - origem_processada: str
    - destinos: list de str (pode ter 1 ou 2 destinos)
    """
    if not nome_lt or pd.isna(nome_lt):
        return None, None

    try:
        # 1. Separar origem e destinos
        partes = nome_lt.split(' - ')
        if len(partes) < 2:
            return None, None

        origem = partes[0]
        destinos = partes[1:]

        # 2. Processar origem: remover tudo antes do kV (incluindo kV)
        indice_kv = origem.rfind('kV')
        origem_processada = origem[indice_kv + 2:].strip() if indice_kv != -1 else origem.strip()

        # 3. Processar cada destino
        destinos_processados = []
        for destino in destinos:
            # Remove (CD) ou similar
            destino = re.sub(r'\(.*?\)', '', destino).strip()

            # Remove conteúdo após a última vírgula (se houver)
            if ',' in destino:
                destino = destino.split(',')[0].strip()

            # Remove C e números no final (ex: C2)
            destino = re.sub(r' C\d+$', '', destino).strip()

            if destino:  # Só adiciona se não for vazio
                destinos_processados.append(destino)

        # Retorna a origem e uma lista de destinos (pode ter 1 ou 2 elementos)
        return origem_processada, destinos_processados if destinos_processados else None

    except Exception as e:
        print(f"Erro ao processar '{nome_lt}': {e}")
        return None, None

def classificar_subestacao(nome):
    """
    Classifica e formata um nome de origem/destino.
    Agora também aceita lista de nomes.
    """
    if isinstance(nome, list):
        return [classificar_subestacao(n) for n in nome if n]

    if not nome or pd.isna(nome):
        return None

    nome = str(nome).strip()
    nome = nome.replace(',', '').strip()

    if any(nome.startswith(prefix) for prefix in ['UTE ', 'UHE ', 'UEE ']):
        return nome
    else:
        if nome and not nome.startswith('SE '):
            return f"SE {nome}"
        return nome

def processar_linhas_e_atualizar_rede(rede_pypsa, gdf_linhas):
    """
    Processa as linhas de transmissão e atualiza a rede PyPSA.
    Versão que trata múltiplos destinos (ramais).
    """

    gdf_linhas['origem_processada'] = None
    gdf_linhas['destinos_processados'] = None  # Agora armazena lista de destinos
    gdf_linhas['conexoes'] = None

    for idx, row in gdf_linhas.iterrows():
        try:
            # Processar nome da LT (agora retorna origem e lista de destinos)
            origem, destinos = processar_nome_lt(row['Nome'])

            # Classificar origem e destinos
            origem_classificada = classificar_subestacao(origem) if origem else None
            destinos_classificados = [classificar_subestacao(d) for d in destinos] if destinos else []

            posicao_origem = row['origem_geom']
            posicao_destino = row['destino_geom']  # Pode precisar ser tratado como lista também

            # Armazenar resultados
            gdf_linhas.at[idx, 'origem_processada'] = origem_classificada
            gdf_linhas.at[idx, 'destinos_processados'] = destinos_classificados

            # Adicionar barramentos à rede
            if origem_classificada:
                tensao_linha = row['Tensao']

                # Adicionar origem se não existir
                if not barramento_existe(rede_pypsa, origem_classificada, tensao_linha, posicao_origem):
                    adicionar_barramento(rede_pypsa, origem_classificada, tensao_linha,
                                         is_geradora(origem_classificada), posicao_origem)

                # Adicionar cada destino
                for i, destino_classificado in enumerate(destinos_classificados):
                    if destino_classificado:
                        # Tratar posição - assumindo que posicao_destino é uma lista quando há múltiplos destinos
                        pos_dest = posicao_destino[i] if isinstance(posicao_destino, (list, tuple)) else posicao_destino

                        if not barramento_existe(rede_pypsa, destino_classificado, tensao_linha, pos_dest):
                            adicionar_barramento(rede_pypsa, destino_classificado, tensao_linha,
                                                 is_geradora(destino_classificado), posicao_destino)
            conexoes = determinar_conexoes_lt(network, origem_classificada, destinos_classificados, row['Tensao'])
            gdf_linhas.at[idx, 'conexoes'] = conexoes

            if conexoes:
                comprimento_total = float(row['Extensao']) if 'Extensao' in row else 1.0

                # Para ramais, dividir o comprimento proporcionalmente
                if len(conexoes) > 1:
                    comprimento_por_trecho = comprimento_total / len(conexoes)
                else:
                    comprimento_por_trecho = comprimento_total

                for i, (bus0, bus1) in enumerate(conexoes):
                    nome_linha = f"{row['Nome']}_trecho_{i + 1}" if len(conexoes) > 1 else row['Nome']

                    try:
                        inserir_lt(network,nome_linha,bus0,bus1,tensao_linha,comprimento_por_trecho)
                    except Exception as e:
                        print(f"Erro ao adicionar linha {nome_linha}: {e}")
                        continue

        except Exception as e:
            print(f"Erro ao processar linha {idx} ('{row['Nome']}'): {e}")
            continue

    network.buses.to_excel('SEs Inseridas.xlsx')
    network.loads.to_excel('Cargas Inseridas.xlsx')
    network.lines.to_excel('linhas inseridas.xlsx')

    return gdf_linhas, rede_pypsa

def determinar_conexoes_lt(network, origem, destinos, tensao):
    """
    Determina os pares de barramentos que uma linha de transmissão deve conectar.

    Args:
        network: Rede PyPSA
        origem: Nome da subestação de origem (str)
        destinos: Lista de nomes de subestações de destino (list[str])
        tensao: Tensão nominal da linha (float)

    Returns:
        Lista de tuplas (barramento_origem, barramento_destino) que devem ser conectados
    """
    conexoes = []

    # Verifica se temos destinos válidos
    if not destinos or len(destinos) == 0:
        return conexoes

    # Obtém todos os barramentos da rede
    buses = network.buses

    # 1. Encontra o barramento de origem correto
    barramento_origem = None

    # Primeiro tenta encontrar por nome e tensão
    mask_origem = (buses['substation'] == origem) & (buses['v_nom'] == tensao)
    if mask_origem.any():
        barramento_origem = buses[mask_origem].index[0]

    # Se não encontrou, tenta encontrar por proximidade geográfica (se implementado)
    # ...

    if barramento_origem is None:
        print(f"Barramento de origem não encontrado: {origem} {tensao}kV")
        return conexoes

    # 2. Para cada destino, encontra o barramento correspondente
    barramentos_destino = []

    for destino in destinos:
        mask_destino = (buses['substation'] == destino) & (buses['v_nom'] == tensao)
        if mask_destino.any():
            barramentos_destino.append(buses[mask_destino].index[0])
        else:
            print(f"Barramento de destino não encontrado: {destino} {tensao}kV")
            barramentos_destino.append(None)

    # 3. Determina os pares de conexão conforme o número de destinos
    if len(barramentos_destino) == 1:
        # Caso simples: origem -> destino1
        if barramentos_destino[0] is not None:
            conexoes.append((barramento_origem, barramentos_destino[0]))
    else:
        # Caso ramal: origem -> destino1 -> destino2
        for i in range(len(barramentos_destino)):
            if i == 0:
                # Primeira conexão: origem -> destino1
                if barramentos_destino[i] is not None:
                    conexoes.append((barramento_origem, barramentos_destino[i]))
            else:
                # Conexões subsequentes: destino anterior -> destino atual
                if barramentos_destino[i - 1] is not None and barramentos_destino[i] is not None:
                    conexoes.append((barramentos_destino[i - 1], barramentos_destino[i]))

    return conexoes

def barramento_existe(rede, nome, v_nom, ponto_referencia=None, distancia_maxima=0.00001):
    """
    Verifica se um barramento existe, agora suportando lista de nomes.
    """
    if isinstance(nome, list):
        return any(barramento_existe(rede, n, v_nom, ponto_referencia, distancia_maxima) for n in nome)

    buses = rede.buses

    if is_geradora(nome): return False

    # Verificação por nome e tensão
    if nome is not None:
        existe_por_nome = ((buses['substation'] == nome) & (buses['v_nom'] == v_nom)).any()
        if existe_por_nome:
            return True

    # Verificação por proximidade geográfica
    if ponto_referencia is not None and isinstance(ponto_referencia, Point):
        try:
            coords = []
            indices_validos = []

            for idx, geom in buses.geometry.items():
                if isinstance(geom, Point):
                    coords.append([geom.x, geom.y])
                    indices_validos.append(idx)

            if not coords:
                return False

            coords_array = np.array(coords)
            ponto_alvo = np.array([ponto_referencia.x, ponto_referencia.y])
            distancias = np.linalg.norm(coords_array - ponto_alvo, axis=1)
            v_nom_filtrado = buses.loc[indices_validos, 'v_nom'].values

            return ((distancias < distancia_maxima) & (v_nom_filtrado == v_nom)).any()

        except Exception as e:
            print(f"Erro na verificação geográfica: {e}")
            return False

    return False

def is_geradora(nome):
    """Verifica se o nome corresponde a uma geradora"""
    return nome.startswith(('UTE ', 'UHE ', 'UEE '))

def adicionar_barramento(rede, nome, v_nom, is_geradora, posicao):
    """Adiciona um barramento à rede PyPSA"""
    # Adiciona o barramento
    rede.add("Bus",
             name=nome,
             v_nom=v_nom,
             v_mag_pu_set = 0.98,
             substation=nome,  # Armazena apenas o nome base
             geometry=posicao)

    # Se for SE (não geradora) e não existir barramento base, adiciona transformador
    if not is_geradora:
        # Nome do barramento base (sem nível de tensão)
        nome_base = nome.replace(f" {int(v_nom)}", "").strip()

        # Verifica se existe barramento base
        if not barramento_existe(rede, nome_base, v_nom):
            # Adiciona barramento base
            rede.add("Bus",
                     name=nome_base,
                     v_nom=v_nom,
                     v_mag_pu_set=0.98,
                     substation=nome,  # Armazena apenas o nome base
                     geometry=posicao
                     )

            # Adiciona transformador entre barramentos
            rede.add("Transformer",
                     name=f"Trafo {nome_base}-{int(v_nom)}",
                     bus0=nome_base,
                     bus1=nome,
                     model="t",
                     r=0.005,
                     x=0.1,
                     b=0.05,
                     g=0)

def inserir_lt(rede, nome, origem, destino, v_nom, extensao):
    if v_nom == 230:
        rede.add('Line',
                 name=nome,
                 bus0=origem,
                 bus1=destino,
                 s_nom=10000,
                 x=0.35 * extensao,
                 r=0.05 * extensao,
                 g=0,  # Desprezível
                 b=3.2e-6 * extensao,
                 length= extensao,
                 num_parallel=5)
    elif v_nom== 345:
        rede.add('Line',
                 name=nome,
                 bus0=origem,
                 bus1=destino,
                 s_nom=10000,
                 x=0.3 * extensao,
                 r=0.04 * extensao,
                 g=0,  # Desprezível
                 b=3.8e-6 * extensao,
                 length=extensao,
                 num_parallel=5)
    elif v_nom==440:
        rede.add('Line',
                 name=nome,
                 bus0=origem,
                 bus1=destino,
                 s_nom=10000,
                 x=0.25 * extensao,
                 r=0.03 * extensao,
                 g=0,  # Desprezível
                 b=4.5e-6 * extensao,
                 length=extensao,
                 num_parallel=5)
    elif v_nom==500:
        rede.add('Line',
                 name=nome,
                 bus0=origem,
                 bus1=destino,
                 s_nom=10000,
                 x=0.2 * extensao,
                 r=0.015 * extensao,
                 g=0,  # Desprezível
                 b=5.0e-6 * extensao,
                 length=extensao,
                 num_parallel=5)
    elif v_nom==525:
        rede.add('Line',
                 name=nome,
                 bus0=origem,
                 bus1=destino,
                 s_nom=10000,
                 x=0.18 * extensao,
                 r=0.012 * extensao,
                 g=0,  # Desprezível
                 b=5.3e-6 * extensao,
                 length=extensao,
                 num_parallel=5)
    else:
        rede.add('Line',
                 name=nome,
                 bus0=origem,
                 bus1=destino,
                 s_nom=10000,
                 x=0.000001 * extensao,
                 r=0.0022 * extensao,
                 g=0,  # Desprezível
                 b=1.0e-6 * extensao,
                 length=extensao,
                 num_parallel=5)

def conectar_geradores_a_rede(rede_pypsa, gdf_geradores):
    """
    Conecta geradores à rede PyPSA usando barramentos existentes.

    Parâmetros:
    ----------
    rede_pypsa : pypsa.Network
        Rede PyPSA existente
    gdf_geradores : geopandas.GeoDataFrame
        Deve conter colunas:
        - nome: Nome do gerador
        - geometry: Posição geográfica (Point)
        - fonte_original: Tipo (UHE, UTE, UEE)
        - potencia: Potência nominal (MW)

    Retorna:
    -------
    pypsa.Network
        Rede com geradores adicionados
    """

    # Verificação de colunas obrigatórias
    required_cols = {'nome', 'geometry', 'fonte_original', 'potencia'}
    if not required_cols.issubset(gdf_geradores.columns):
        missing = required_cols - set(gdf_geradores.columns)
        raise ValueError(f"Faltam colunas: {missing}")

    # Filtrar apenas UHE, UTE, UEE
    tipos_validos = ['UHE', 'UTE', 'UEE']
    gdf_filtrado = gdf_geradores[gdf_geradores['fonte_original'].isin(tipos_validos)].copy()

    if gdf_filtrado.empty:
        print("Nenhum gerador do tipo UHE, UTE ou UEE encontrado.")
        return rede_pypsa

    # Pré-processamento: Criar lista de barramentos com geometria
    buses_with_geom = rede_pypsa.buses[rede_pypsa.buses.geometry.notnull()].copy()

    for idx, gerador in gdf_filtrado.iterrows():
        try:
            nome = str(gerador['nome']).strip()
            fonte = str(gerador['fonte_original']).strip()
            potencia = float(gerador['potencia'])
            geom_gerador = gerador['geometry']

            # 1. Tentar encontrar barramento com nome exato do gerador
            bus_candidato = None

            if nome in rede_pypsa.buses.index:
                bus_candidato = nome
            else:
                # 2. Encontrar barramento mais próximo
                if isinstance(geom_gerador, Point) and not buses_with_geom.empty:
                    # Calcular distâncias para todos os barramentos
                    distancias = buses_with_geom.geometry.apply(
                        lambda x: geom_gerador.distance(x) if isinstance(x, Point) else float('inf'))

                    if not distancias.empty:
                        idx_proximo = distancias.idxmin()
                        if distancias[idx_proximo] < 10:  # 10km de tolerância
                            bus_candidato = idx_proximo

            if bus_candidato is None:
                print(f"Nenhum barramento adequado encontrado para {nome} - gerador não conectado")
                continue

            # Parâmetros do gerador
            generator_params = {
                'name': nome,
                'bus': bus_candidato,
                'p_nom': potencia * 0.92,
                'q_set': potencia * 0.39,
                'control': "Slack" if "Paulo Afonso I" in nome else "PV",
                'type': fonte,
                'p_max_pu': 1
            }

            # Configurações especiais para eólicas (UEE)
            if fonte == 'UEE':
                generator_params.update({
                    'p_min_pu': 1,
                    'p_max_pu': 1
                })

            # Adicionar à rede
            rede_pypsa.add("Generator", **generator_params)

        except Exception as e:
            print(f"Erro ao conectar gerador {nome}: {str(e)}")
            continue

    print(f"Processamento concluído. {len(gdf_filtrado)} geradores processados.")
    print(f"Total de geradores na rede: {len(rede_pypsa.generators)}")

    return rede_pypsa

network= pypsa.Network()
subestacoes = carregar_SEs()
linhas = carregar_LTs()
geracao = carregar_Geracao()
linhas = extrair_pontos_referencia(linhas)
network = add_substations_to_pypsa(network, subestacoes)

gdf_processado, network = processar_linhas_e_atualizar_rede(network, linhas)
network = conectar_geradores_a_rede(network, geracao)


def diagnose_network(network):
    """Função para diagnóstico completo da rede"""

    print("\n" + "=" * 50)
    print("DIAGNÓSTICO COMPLETO DA REDE")
    print("=" * 50)

    # 1. Informações básicas
    print("\n[INFORMAÇÕES BÁSICAS]")
    print(f"Total de barramentos: {len(network.buses)}")
    print(f"Total de linhas: {len(network.lines)}")
    print(f"Total de geradores: {len(network.generators)}")
    print(f"Total de cargas: {len(network.loads)}")

    # 2. Verificar Slack bus
    print("\n[SLACK BUS]")
    slack_buses = network.generators[network.generators.control == 'Slack']
    if len(slack_buses) == 0:
        print("ERRO: Nenhuma slack bus definida!")
    else:
        print(f"Slack bus encontrada no gerador: {slack_buses.index[0]}")

    # 3. Verificar linhas
    print("\n[LINHAS]")
    print("Primeiras 5 linhas:")
    print(network.lines[['bus0', 'bus1', 'x', 'r', 's_nom']].head())

    # Verificar linhas problemáticas
    zero_x_lines = network.lines[network.lines.x == 0]
    if not zero_x_lines.empty:
        print(f"\nAVISO: {len(zero_x_lines)} linha(s) com reatância zero:")
        print(zero_x_lines.index.tolist())

    # 4. Verificar cargas (versão corrigida)
    print("\n[ANÁLISE DE CARGAS]")

    # Verifica se existem cargas definidas
    if len(network.loads) == 0:
        print("AVISO CRÍTICO: Nenhuma carga definida no sistema!")
    else:
        # Verifica se os valores temporais das cargas estão disponíveis
        if hasattr(network, 'loads_t') and 'p' in network.loads_t:
            print("\nValores ativos das cargas (MW):")
            # Converter para MW e mostrar todas as cargas (não apenas as primeiras)
            cargas_mw = network.loads_t.p.loc["now"] / 1e6
            print(cargas_mw.to_string())  # to_string() mostra todas as linhas

            total_carga = cargas_mw.sum()
            print(f"\nCarga total do sistema: {total_carga:.2f} MW")

            # Verificar se há cargas zeradas
            cargas_zeradas = cargas_mw[cargas_mw == 0]
            if not cargas_zeradas.empty:
                print(f"\nAVISO: {len(cargas_zeradas)} carga(s) com valor zero:")
                print(cargas_zeradas.index.tolist())
        else:
            print("Dados temporais de carga não disponíveis (network.loads_t.p)")

        # Mostrar informações estáticas das cargas
        print("\nInformações estáticas das cargas:")
        print(network.loads[['bus', 'p_set']].head())  # Mostra as primeiras 5

    # 5. Verificar geradores
    print("\n[GERADORES]")
    print("Primeiros 5 geradores:")
    print(network.generators[['bus', 'p_nom', 'control']].head())

    if hasattr(network, 'generators_t'):
        total_gen = network.generators_t.p.loc["now"].sum() / 1e6
        print(f"\nGeração total do sistema: {total_gen:.2f} MW")

    # 6. Verificar conectividade
    print("\n[CONECTIVIDADE]")
    try:
        if len(network.sub_networks) > 1:
            print(f"AVISO: Rede contém {len(network.sub_networks)} sub-redes desconectadas!")
        else:
            print("Rede está totalmente conectada")
    except AttributeError:
        print("Não foi possível verificar conectividade (versão mais recente do PyPSA)")

    print("\n" + "=" * 50)
    print("FIM DO DIAGNÓSTICO")
    print("=" * 50)

# 1. Copiar valores estáticos para temporais (se ainda não feito)
if not hasattr(network, 'generators_t'):
    network.generators_t = {}
if not hasattr(network, 'loads_t'):
    network.loads_t = {}

# 2. Definir valores iniciais para o snapshot atual
network.generators_t['p'] = pd.DataFrame(index=network.snapshots,
                                       columns=network.generators.index)
network.loads_t['p'] = pd.DataFrame(index=network.snapshots,
                                  columns=network.loads.index)

# 3. Preencher com valores base (do projeto ou operação)
network.generators_t['p'].loc['now'] = network.generators.p_nom  # ou outro valor
network.loads_t['p'].loc['now'] = network.loads.p_set  # Usando os valores estáticos

# 4. Verificar novamente
print("\nValores temporais após inicialização:")
print("Geração (MW):\n", network.generators_t['p'].loc['now'])
print("\nCargas (MW):\n", network.loads_t['p'].loc['now'])


# Converter todos os dados numéricos para float64 explicitamente
network.lines['x'] = network.lines['x'].astype('float64')
network.lines['r'] = network.lines['r'].astype('float64')
network.lines['s_nom'] = network.lines['s_nom'].astype('float64')

network.buses['v_nom'] = network.buses['v_nom'].astype('float64')

# Verificar e converter dados dos geradores
network.generators['p_nom'] = network.generators['p_nom'].astype('float64')
network.generators_t['p'] = network.generators_t['p'].astype('float64')

# Verificar e converter dados das cargas
network.loads['p_set'] = network.loads['p_set'].astype('float64')
network.loads_t['p'] = network.loads_t['p'].astype('float64')


# Obter todos os barramentos
all_buses = set(network.buses.index)

# Obter barramentos conectados (que aparecem em linhas ou transformadores)
connected_buses = set()
connected_buses.update(network.lines.bus0)
connected_buses.update(network.lines.bus1)
connected_buses.update(network.transformers.bus0)
connected_buses.update(network.transformers.bus1)

# Encontrar barramentos desconectados
disconnected_buses = all_buses - connected_buses
# Opção para remover barramentos desconectados (se desejado)
if disconnected_buses:
    network.mremove("Bus", list(disconnected_buses))
    print(f"Barramentos desconectados removidos: {len(disconnected_buses)}")

# Obter todos os barramentos
all_buses = set(network.buses.index)

# Obter barramentos conectados (que aparecem em linhas ou transformadores)
connected_buses = set()
connected_buses.update(network.lines.bus0)
connected_buses.update(network.lines.bus1)
connected_buses.update(network.transformers.bus0)
connected_buses.update(network.transformers.bus1)

# Encontrar barramentos desconectados
disconnected_buses = all_buses - connected_buses

db=pd.DataFrame(disconnected_buses)
db.to_excel('SEs não conectadas.xlsx')

try:
    network.lpf()  # Tenta o fluxo de potência
except Exception as e:
    print(f"Falha no fluxo: {e}")

    # Verifica tensões (se o cálculo parcial foi armazenado)
    if hasattr(network, 'buses_t'):
        df_log = network.buses_t.v_mag_pu
        df_log=df_log.T


    # Verifica linhas com fluxos extremos
    if hasattr(network, 'lines_t'):
        df_log_PA = network.lines_t.p0
        df_log_PA = df_log_PA.T
    resultado = pd.concat([df_log_PA, df_log], axis=1)
    resultado.to_excel('log de dados.xlsx')
# 8. Analisar resultados

if not network.lines_t.p0.empty:
    utilizacao_linhas = (abs(network.lines_t.p0.loc["now"]) / network.lines.s_nom * 100).mean()
    print(f"\nUtilização média das linhas: {utilizacao_linhas:.2f}%")
# 3. Acessar os resultados (exemplo):
if network.lines_t.p0.empty:
    print("Aviso: Nenhum resultado de fluxo calculado!")
else:
    # Criar DataFrames separadamente para maior clareza
    df_linhas = pd.DataFrame({
        'Nome': network.lines.index,
        'De': network.lines.bus0,
        'Para': network.lines.bus1,
        'Fluxo_MW': network.lines_t.p0.loc["now"],  # MW
        'Capacidade_MW': network.lines.s_nom,  # MW
        'Utilizacao_%': (abs(network.lines_t.p0.loc["now"]) / network.lines.s_nom * 100)
    })

    df_geradores = pd.DataFrame({
        'Nome': network.generators.index,
        'Barramento': network.generators.bus,
        'Geracao_MW': network.generators_t.p.loc["now"],  # MW
        'Capacidade_MW': network.generators.p_nom # MW
    })

    df_cargas = pd.DataFrame({
        'Nome': network.loads.index,
        'Barramento': network.loads.bus,
        'Carga_MW': network.loads_t.p.loc["now"] # MW
    })
    # Novo DataFrame para barramentos e tensões
    df_barramentos = pd.DataFrame({
        'Barramento': network.buses.index,
        'Tensao_pu': network.buses_t.v_mag_pu.loc["now"],
        'Angulo_graus': network.buses_t.v_ang.loc["now"] * 180 / 3.14159  # Convertendo radianos para graus
    })

    # Salvar em Excel
    with pd.ExcelWriter('resultados_detalhados.xlsx') as writer:
        df_linhas.to_excel(writer, sheet_name='Linhas', index=False)
        df_geradores.to_excel(writer, sheet_name='Geradores', index=False)
        df_cargas.to_excel(writer, sheet_name='Cargas', index=False)
        df_barramentos.to_excel(writer, sheet_name='Barramentos', index=False)

    print("\nResultados detalhados salvos em 'resultados_detalhados.xlsx'")

