import pypsa
import pandas as pd
import numpy as np
import geopandas as gpd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from shapely.geometry import Point, LineString, MultiLineString
import re


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

        # Filtra apenas geradores com potência >= 10 MW (APÓS a conversão)
        gdf_final = gdf_final[gdf_final['potencia'] >= 10]

        # Adiciona coluna de subsistema baseado nas coordenadas
        gdf_final['subsistema'] = None

        # Classifica cada gerador em um subsistema
        for idx, row in gdf_final.iterrows():
            if isinstance(row.geometry, Point):
                lat = row.geometry.y
                lon = row.geometry.x

                if lat < -20:  # Sul
                    gdf_final.at[idx, 'subsistema'] = 'S'
                elif -20 <= lat <= -10 and lon > -50:  # Sudeste/Centro-Oeste
                    gdf_final.at[idx, 'subsistema'] = 'SE/CO'
                elif lat > -10 and lon > -40:  # Nordeste
                    gdf_final.at[idx, 'subsistema'] = 'NE'
                else:  # Norte
                    gdf_final.at[idx, 'subsistema'] = 'N'
            else:
                print(f"Gerador {idx} não tem geometria Point válida")

        gdf_final.to_excel('geração.xlsx')
        return gdf_final

    except Exception as e:
        print(f"Erro ao carregar arquivos shapefile: {e}")
        raise

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

    if any(nome.startswith(prefix) for prefix in ['PCH', 'CGH', 'UEE', 'UFV', 'UHE', 'UTE']):
        return nome
    else:
        if nome and not nome.startswith('SE '):
            return f"SE {nome}"
        return nome

def processar_linhas_e_atualizar_rede(rede_pypsa, gdf_linhas):
    """
    Processa as linhas de transmissão e atualiza a rede PyPSA com visualização passo a passo.
    Versão que plota cada linha adicionada para verificar as conexões.
    """

    gdf_linhas['origem_processada'] = None
    gdf_linhas['destinos_processados'] = None
    gdf_linhas['conexoes'] = None



    for idx, row in gdf_linhas.iterrows():
        try:
            # Processar nome da LT
            origem, destinos = processar_nome_lt(row['Nome'])
            origem_classificada = classificar_subestacao(origem) if origem else None
            destinos_classificados = [classificar_subestacao(d) for d in destinos] if destinos else []

            # Armazenar resultados
            gdf_linhas.at[idx, 'origem_processada'] = origem_classificada
            gdf_linhas.at[idx, 'destinos_processados'] = destinos_classificados

            # Adicionar barramentos
            if origem_classificada:
                tensao_linha = row['Tensao']
                posicao_origem = row['origem_geom']
                posicao_destino = row['destino_geom']

                if not barramento_existe(rede_pypsa, origem_classificada, tensao_linha, posicao_origem):
                    adicionar_barramento(rede_pypsa, origem_classificada, tensao_linha,
                                         is_geradora(origem_classificada), posicao_origem)

                for i, destino_classificado in enumerate(destinos_classificados):
                    if destino_classificado:
                        pos_dest = posicao_destino[i] if isinstance(posicao_destino, (list, tuple)) else posicao_destino
                        if not barramento_existe(rede_pypsa, destino_classificado, tensao_linha, pos_dest):
                            adicionar_barramento(rede_pypsa, destino_classificado, tensao_linha,
                                                 is_geradora(destino_classificado), pos_dest)

            # Determinar conexões
            conexoes = determinar_conexoes_lt(rede_pypsa, origem_classificada, destinos_classificados, row['Tensao'])
            gdf_linhas.at[idx, 'conexoes'] = conexoes

            if conexoes:
                comprimento_total = float(row['Extensao']) if 'Extensao' in row else 1.0
                comprimento_por_trecho = comprimento_total / len(conexoes) if len(conexoes) > 1 else comprimento_total

                for i, (bus0, bus1) in enumerate(conexoes):
                    nome_linha = f"{row['Nome']}_trecho_{i + 1}" if len(conexoes) > 1 else row['Nome']

                    # Adicionar linha à rede
                    inserir_lt(rede_pypsa, nome_linha, bus0, bus1, row['Tensao'], comprimento_por_trecho)



        except Exception as e:
            print(f"Erro ao processar linha {idx} ('{row['Nome']}'): {e}")
            continue


    # Salvar resultados
    rede_pypsa.buses.to_excel('SEs_Inseridas.xlsx')
    rede_pypsa.loads.to_excel('Cargas_Inseridas.xlsx')
    rede_pypsa.lines.to_excel('linhas_inseridas.xlsx')
    gdf_linhas.to_excel('linhas_processadas.xlsx')

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
    return False

def is_geradora(nome):
    """Verifica se o nome corresponde a uma geradora (UEE, UHE ou UTE)"""
    if not isinstance(nome, str):
        return False
    return nome.startswith(('UEE', 'UHE', 'UTE'))

def adicionar_barramento(rede, nome, v_nom, is_geradora, posicao):
    """Adiciona um barramento à rede PyPSA"""
    # Adiciona o barramento
    rede.add("Bus",
             name=f"{nome} V_{v_nom}",
             v_nom=v_nom,
             v_mag_pu_set = 1,
             substation=nome,  # Armazena apenas o nome base
             geometry=posicao)

def inserir_lt(rede, nome, origem, destino, v_nom, extensao):
    if v_nom == 230:
        rede.add('Line',
                 name=nome,
                 bus0=origem,
                 bus1=destino,
                 s_nom=10000,
                 v_nom = v_nom,
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
                 v_nom=v_nom,
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
                 v_nom=v_nom,
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
                 v_nom=v_nom,
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
                 v_nom=v_nom,
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
                 v_nom=v_nom,
                 x=0.000001 * extensao,
                 r=0.0022 * extensao,
                 g=0,  # Desprezível
                 b=1.0e-6 * extensao,
                 length=extensao,
                 num_parallel=5)

def conectar_geradores_a_rede(rede_pypsa, gdf_geradores):
    """
    Conecta geradores à rede PyPSA usando barramentos existentes.
    Se já existir gerador no barramento, cria novo barramento e conecta com transformador.

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
    tipos_validos = ['UHE', 'UTE', 'UEE', 'PCH', 'UFV']
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
            subsist = gerador['subsistema']

            # 1. Tentar encontrar barramento onde substation == nome do gerador
            bus_candidato = None

            # Verificar se há barramento com substation igual ao nome do gerador
            if 'substation' in rede_pypsa.buses.columns:
                mask = rede_pypsa.buses['substation'] == nome
                if mask.any():
                    bus_candidato = rede_pypsa.buses[mask].index[0]

            # 2. Se não encontrou, verificar se há barramento com nome exato do gerador
            if bus_candidato is None and nome in rede_pypsa.buses.index:
                bus_candidato = nome

            # 3. Se ainda não encontrou, procurar barramento mais próximo
            if bus_candidato is None and isinstance(geom_gerador, Point) and not buses_with_geom.empty:
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

            # Verificar se já existe gerador neste barramento
            if bus_candidato in rede_pypsa.generators.bus.values:
                print(f"Já existe um gerador no barramento {bus_candidato} - criando novo barramento conectado")

                # Obter informações do barramento original
                bus_original = rede_pypsa.buses.loc[bus_candidato]

                # Criar novo nome para o barramento
                novo_bus_name = f"{bus_candidato}_{nome}"

                # Adicionar novo barramento com mesmas características
                rede_pypsa.add("Bus",
                               name=novo_bus_name,
                               v_nom=bus_original['v_nom'],
                               v_mag_pu_set=1.0,
                               x=bus_original['x'],
                               y=bus_original['y'],
                               substation=bus_original.get('substation', ''),
                               geometry=bus_original.get('geometry', None))

                # Adicionar transformador entre os barramentos
                rede_pypsa.add("Transformer",
                               name=f"Trafo_{bus_candidato}_to_{novo_bus_name}",
                               bus0=bus_candidato,
                               bus1=novo_bus_name,
                               model="t",
                               r=0.0005,
                               x=0.01,
                               b=0.05,
                               g=0)

                # Usar o novo barramento para o gerador
                bus_candidato = novo_bus_name

            # Parâmetros do gerador
            generator_params = {
                'name': nome,
                'bus': bus_candidato,
                'p_nom': potencia * 0.92,  # Considerando fator de capacidade
                'q_set': potencia * 0.39,  # Valor de exemplo para reativo
                'control': "Slack" if "Paulo Afonso I" in nome else "PV",
                'type': fonte,
                'p_max_pu': 1,
                'subsistema': subsist
            }

            # Configurações especiais para eólicas (UEE)
            if fonte == 'UEE':
                generator_params.update({
                    'p_min_pu': 0,  # Eólicas podem ter geração zero
                    'p_max_pu': 1
                })

            # Adicionar à rede
            rede_pypsa.add("Generator", **generator_params)
            print(f"Gerador {nome} conectado ao barramento {bus_candidato}")

        except Exception as e:
            print(f"Erro ao conectar gerador {nome}: {str(e)}")
            continue

    df_geradores=pd.DataFrame(rede_pypsa.generators)
    df_geradores.to_excel('geradores.xlsx')

    print(f"Processamento concluído. {len(gdf_filtrado)} geradores processados.")
    print(f"Total de geradores na rede: {len(rede_pypsa.generators)}")

    return rede_pypsa

def conectar_barramentos_mesma_subestacao(rede_pypsa):
    """
    Conecta barramentos da mesma subestação com transformadores.

    Parâmetros:
    ----------
    rede_pypsa : pypsa.Network
        Rede PyPSA existente

    Retorna:
    -------
    pypsa.Network
        Rede com transformadores adicionados entre barramentos da mesma subestação
    """

    # Verificar se existe a coluna 'substation' nos barramentos
    if 'substation' not in rede_pypsa.buses.columns:
        print("A rede não possui a coluna 'substation' nos barramentos - nada a conectar")
        return rede_pypsa

    # Agrupar barramentos por subestação
    grupos = rede_pypsa.buses.groupby('substation')

    # Contador de transformadores adicionados
    trafos_adicionados = 0

    for substation_name, group in grupos:
        # Obter lista de barramentos nesta subestação
        barramentos = group.index.tolist()

        # Se houver mais de um barramento na mesma subestação
        if len(barramentos) > 1:
            # Ordenar por tensão nominal (maior para menor)
            barramentos_ordenados = sorted(barramentos,
                                           key=lambda x: rede_pypsa.buses.at[x, 'v_nom'],
                                           reverse=True)

            # Conectar cada barramento ao barramento de tensão mais alta (primeiro da lista)
            barramento_principal = barramentos_ordenados[0]

            for bus in barramentos_ordenados[1:]:
                # Verificar se já existe um transformador conectando esses barramentos
                trafo_existe = False
                for _, trafo in rede_pypsa.transformers.iterrows():
                    if (trafo['bus0'] == barramento_principal and trafo['bus1'] == bus) or \
                            (trafo['bus0'] == bus and trafo['bus1'] == barramento_principal):
                        trafo_existe = True
                        break

                if not trafo_existe:
                    # Criar nome único para o transformador
                    trafo_name = f"Trafo_{substation_name}_{rede_pypsa.buses.at[bus, 'v_nom']}kV"

                    # Adicionar transformador
                    rede_pypsa.add("Transformer",
                                   name=trafo_name,
                                   bus0=barramento_principal,
                                   bus1=bus,
                                   model="t",
                                   r=0.005,
                                   x=0.1,
                                   b=0.05,
                                   g=0)

                    trafos_adicionados += 1
                    print(f"Adicionado transformador {trafo_name} conectando {barramento_principal} e {bus}")

    print(f"Processamento concluído. {trafos_adicionados} transformadores adicionados.")
    return rede_pypsa

def plotar_mapa_e_geracao(rede_pypsa, gdf_geradores, salvar_imagem=True, nome_imagem='mapa_rede_brasileira.png'):
    """
    Plota um mapa do Brasil com linhas de transmissão por tensão e geradoras por tipo,
    e retorna um dataframe com a geração por tipo para cada subsistema.

    Parâmetros:
    ----------
    rede_pypsa : pypsa.Network
        Rede PyPSA com os dados da rede elétrica
    gdf_geradores : geopandas.GeoDataFrame
        GeoDataFrame com os dados dos geradores
    salvar_imagem : bool, opcional
        Se True, salva a imagem do mapa (padrão: True)
    nome_imagem : str, opcional
        Nome do arquivo de imagem a ser salvo (padrão: 'mapa_rede_brasileira.png')

    Retorna:
    -------
    pandas.DataFrame
        DataFrame com a geração agregada por tipo e subsistema
    """

    # Configurar o plot
    plt.figure(figsize=(20, 15))
    ax = plt.gca()

    # Carregar base do Brasil
    try:
        brasil = gpd.read_file(r"C:\Users\pgcs_\PycharmProjects\PyPSA\raw\BR_UF_2021.shp")
        brasil.plot(ax=ax, color='#f0f0f0', edgecolor='black', linewidth=0.5)
    except Exception as e:
        print(f"Erro ao carregar base do Brasil: {e}")
        # Criar um eixo vazio se não conseguir carregar o mapa
        ax.set_xlim(-75, -30)
        ax.set_ylim(-35, 5)

    # 1. Plotar linhas de transmissão por nível de tensão
    if hasattr(rede_pypsa, 'lines'):
        # Definir cores e legendas para cada nível de tensão
        tensoes_cores = {
            230: ('blue', '230 kV'),
            345: ('green', '345 kV'),
            440: ('orange', '440 kV'),
            500: ('red', '500 kV'),
            525: ('purple', '525 kV'),
            'Others': ('gray', 'Others')
        }

        for idx, linha in rede_pypsa.lines.iterrows():
            try:
                # Obter barramentos conectados
                bus0 = rede_pypsa.buses.loc[linha.bus0]
                bus1 = rede_pypsa.buses.loc[linha.bus1]

                # Obter coordenadas
                x0, y0 = bus0.geometry.x, bus0.geometry.y
                x1, y1 = bus1.geometry.x, bus1.geometry.y

                # Determinar cor baseada na tensão nominal
                v_nom = linha.v_nom if hasattr(linha, 'v_nom') else None
                if v_nom in tensoes_cores:
                    cor, legenda = tensoes_cores[v_nom]
                else:
                    cor, legenda = tensoes_cores['outros']

                # Plotar linha
                ax.plot([x0, x1], [y0, y1], color=cor, linewidth=1, alpha=0.7)

            except Exception as e:
                print(f"Erro ao plotar linha {idx}: {e}")

    # 2. Plotar geradoras por tipo
    if gdf_geradores is not None and not gdf_geradores.empty:
        # Definir cores e marcadores para cada tipo de gerador
        geradores_estilos = {
            'UHE': ('blue', '^', 'UHE'),
            'UTE': ('red', 's', 'UTE'),
            'UEE': ('green', 'o', 'UEE'),
            'UFV': ('yellow', '*', 'UFV'),
            'Others': ('gray', 'x', 'Others')
        }

        for idx, gerador in gdf_geradores.iterrows():
            try:
                # Obter geometria do gerador
                if hasattr(gerador, 'geometry'):
                    geom = gerador.geometry
                    x, y = geom.x, geom.y
                else:
                    continue

                # Determinar estilo baseado no tipo
                fonte = gerador.get('fonte_original', 'outros')
                if fonte in geradores_estilos:
                    cor, marcador, legenda = geradores_estilos[fonte]
                else:
                    cor, marcador, legenda = geradores_estilos['outros']

                # Plotar gerador
                ax.scatter(x, y, color=cor, marker=marcador, s=50, alpha=0.8, edgecolors='black')

            except Exception as e:
                print(f"Erro ao plotar gerador {idx}: {e}")

    # 3. Criar legendas
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    # Legendas para linhas de transmissão
    legendas_linhas = [Line2D([0], [0], color=cor, lw=2, label=legenda)
                       for cor, legenda in tensoes_cores.values()]

    # Legendas para geradores
    legendas_geradores = [Line2D([0], [0], color=cor, marker=marker, linestyle='None',
                                 markersize=8, label=legenda)
                          for cor, marker, legenda in geradores_estilos.values()]

    # Adicionar legendas ao plot
    ax.legend(handles=legendas_linhas + legendas_geradores, loc='upper right', fontsize=10)

    # Configurações do gráfico
    ax.set_title('Brazilian Electrical Grid - Transmission and Generation', fontsize=16)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')

    # Salvar imagem se solicitado
    if salvar_imagem:
        plt.savefig(nome_imagem, dpi=300, bbox_inches='tight')
        print(f"Mapa salvo como {nome_imagem}")

    plt.show()

    # 4. Calcular geração por tipo e subsistema
    if hasattr(rede_pypsa, 'generators') and hasattr(rede_pypsa.buses, 'subsistema'):
        # Criar DataFrame com informações dos geradores
        df_geradores = pd.DataFrame(rede_pypsa.generators)

             # Agrupar por tipo e subsistema
        if 'type' in df_geradores.columns and 'p_nom' in df_geradores.columns:
            df_geracao = df_geradores.groupby(['type', 'subsistema'])['p_nom'].sum().unstack().fillna(0)

            # Adicionar totais
            df_geracao['Total'] = df_geracao.sum(axis=1)
            df_geracao.loc['Total'] = df_geracao.sum(axis=0)

            print("\nGeração por tipo e subsistema (MW):")
            print(df_geracao)

            return df_geracao
        else:
            print("Dados incompletos para calcular geração por subsistema")
            return None
    else:
        print("Rede não contém informações necessárias para cálculo de geração por subsistema")
        return None


network= pypsa.Network()
substation = {
    'name': 'SE Abdon Batista',
    'geometry': Point(-51.072193056999936, -27.57180735999998),
    'subsistema': 'S'
}
network.add("Bus",
            name='SE Abdon Batista V_525',
            v_nom=525,
            v_mag_pu_set=1,
            geometry=substation['geometry'],
            substation=substation['name'],
            subsistema=substation['subsistema'])
linhas = carregar_LTs()
geracao = carregar_Geracao()
linhas = extrair_pontos_referencia(linhas)

gdf_processado, network = processar_linhas_e_atualizar_rede(network, linhas)
network = conectar_barramentos_mesma_subestacao(network)

network = conectar_geradores_a_rede(network, geracao)
df_geracao = plotar_mapa_e_geracao(network, geracao)

df_geracao.to_excel('dados por subsistema.xlsx')

