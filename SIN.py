import pypsa
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point, LineString
import re

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

        return linhas_completas

    except Exception as e:
        print(f"Erro ao carregar arquivos shapefile: {e}")
        exit()

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

                if op == 'Base_Existente':
                    gdf['ini_oper'] = gdf['ini_oper'].replace(['-', 0.0], np.nan)

                # Colunas para remover (usando nomes padronizados)
                cols_to_drop = ['Shape_STLe', 'created_da', 'created_us', 'last_edite', 'last_edi_1', 'combust', 'COMBUST', 'Rio', ' Leilao', 'ano_prev', 'CEG', 'Leilão', 'leilao']
                gdf = gdf.drop(columns=[col for col in cols_to_drop if col in gdf.columns], errors='ignore')

                geradores.append(gdf)

        # Concatena todos os DataFrames
        gdf_final = pd.concat(geradores, axis=0).reset_index(drop=True)
        gdf_final['potencia'] = gdf_final['potencia'] / 1e3  # Convertendo para MW

        return gdf_final

    except Exception as e:
        print(f"Erro ao carregar arquivos shapefile: {e}")
        raise  # Melhor que exit() para permitir tratamento do erro


def conectar_linhas_subestacoes(linhas, subestacoes, geradores):
    """
    Conecta as linhas de transmissão às subestações de origem e destino

    Args:
        linhas: GeoDataFrame com as linhas de transmissão
        subestacoes: GeoDataFrame com as subestações
        geradores: GeoDataFrame com as unidades de geração

    Returns:
        GeoDataFrame com as linhas de transmissão conectadas às SEs
    """
    try:
        # 1. Converter para CRS projetado adequado (ex: UTM 23S para Brasil - EPSG:31983)
        CRS_PROJETADO = "EPSG:31983"  # UTM 23S (SIRGAS 2000)

        # Verificar e converter CRS
        if subestacoes.crs is None:
            subestacoes = subestacoes.set_crs("EPSG:4326")
        subestacoes = subestacoes.to_crs(CRS_PROJETADO)

        linhas = linhas.to_crs(CRS_PROJETADO)
        geradores = geradores.to_crs(CRS_PROJETADO)

        # 2. Padronizar colunas
        def padronizar_colunas(gdf):
            mapeamento = {
                'nome': ['nome', 'NOME', 'Nome', 'name', 'NAME'],
                'geometry': ['geometry', 'geom', 'GEOMETRY']
            }
            for padrao, alternativas in mapeamento.items():
                for alt in alternativas:
                    if alt in gdf.columns:
                        gdf = gdf.rename(columns={alt: padrao})
                        break
            return gdf

        subestacoes = padronizar_colunas(subestacoes)
        geradores = padronizar_colunas(geradores)

        # 3. Combinar pontos de conexão
        pontos_conexao = pd.concat([
            subestacoes[['nome', 'geometry']],
            geradores[['nome', 'geometry']]
        ], ignore_index=True)

        # 4. Função para extrair pontos de extremidade
        def extrair_pontos_extremidade(geom):
            if geom.geom_type == 'LineString':
                return Point(geom.coords[0]), Point(geom.coords[-1])
            elif geom.geom_type == 'MultiLineString':
                # Pega o primeiro ponto da primeira linha e último ponto da última linha
                all_coords = []
                for line in geom.geoms:
                    all_coords.extend(line.coords)
                return Point(all_coords[0]), Point(all_coords[-1])
            else:
                raise ValueError(f"Tipo de geometria não suportado: {geom.geom_type}")

        # 5. Encontrar conexões
        linhas_conectadas = linhas.copy()

        # Listas para armazenar resultados
        se_origem_list = []
        se_destino_list = []
        dist_origem_list = []
        dist_destino_list = []

        for idx, linha in linhas.iterrows():
            try:
                pt_origem, pt_destino = extrair_pontos_extremidade(linha.geometry)

                # Calcular distâncias
                dist_origem = pontos_conexao.distance(pt_origem)
                dist_destino = pontos_conexao.distance(pt_destino)

                # Encontrar pontos mais próximos
                idx_origem = dist_origem.idxmin()
                idx_destino = dist_destino.idxmin()

                # Obter nomes
                nome_origem = pontos_conexao.iloc[idx_origem].get('nome',
                                                                  pontos_conexao.iloc[idx_origem].get('nome_gerador',
                                                                                                      None))
                nome_destino = pontos_conexao.iloc[idx_destino].get('nome',
                                                                    pontos_conexao.iloc[idx_destino].get('nome_gerador',
                                                                                                         None))

                # Armazenar resultados
                se_origem_list.append(nome_origem)
                se_destino_list.append(nome_destino)
                dist_origem_list.append(dist_origem.min())
                dist_destino_list.append(dist_destino.min())

            except Exception as e:
                print(f"Erro na linha {idx}: {str(e)}")
                se_origem_list.append(None)
                se_destino_list.append(None)
                dist_origem_list.append(None)
                dist_destino_list.append(None)

        # Adicionar colunas ao DataFrame
        linhas_conectadas['SE_origem'] = se_origem_list
        linhas_conectadas['SE_destino'] = se_destino_list
        linhas_conectadas['dist_origem'] = dist_origem_list
        linhas_conectadas['dist_destino'] = dist_destino_list

        # 6. Verificar conexões problemáticas
        limite_dist = 100  # 100 metros (em CRS projetado)
        problemas = linhas_conectadas[
            (linhas_conectadas['SE_origem'].isna()) |
            (linhas_conectadas['SE_destino'].isna()) |
            (linhas_conectadas['dist_origem'] > limite_dist) |
            (linhas_conectadas['dist_destino'] > limite_dist)
            ]

        if not problemas.empty:
            print(f"\nAviso: {len(problemas)} linhas com problemas de conexão")
            print(f"Distância máxima de origem: {problemas['dist_origem'].max():.2f} metros")
            print(f"Distância máxima de destino: {problemas['dist_destino'].max():.2f} metros")

        return linhas_conectadas.to_crs("EPSG:4326")  # Retornar para WGS84 se necessário

    except Exception as e:
        print(f"Erro ao conectar linhas: {str(e)}")
        raise


import pypsa
import numpy as np
import re


def adicionar_subestacoes(network, subestacoes_gdf):
    """
    Adiciona subestações como barramentos na rede PyPSA, tratando subestações
    com múltiplas tensões (elevadoras/rebaixadoras).

    Args:
        network: Objeto PyPSA Network
        subestacoes_gdf: GeoDataFrame com as subestações
                         Deve conter colunas 'nome' e 'tensao'
                         Exemplo de tensão: "138/69/13.8" (kV)
    """
    # Expressão regular para extrair tensões
    padrao_tensao = re.compile(r'(\d+\.?\d*)\/?')

    for _, sub in subestacoes_gdf.iterrows():
        nome_se = sub['Nome']
        tensoes = padrao_tensao.findall(str(sub['Tensao']))

        # Converter para float e multiplicar por kV -> V
        tensoes_kV = [float(t) for t in tensoes if t]
        tensoes_V = [t * 1e3 for t in tensoes_kV]

        if not tensoes_V:
            print(f"Aviso: Subestação {nome_se} sem tensão definida. Pulando.")
            continue

        # Caso de subestação simples (uma tensão)
        if len(tensoes_V) == 1:
            if nome_se not in network.buses.index:
                network.add("Bus",
                            name=nome_se,
                            v_nom=tensoes_V[0],
                            x=sub.geometry.x,
                            y=sub.geometry.y)

        # Caso de subestação com múltiplas tensões
        else:
            for i, tensao in enumerate(tensoes_V):
                # Nome do barramento: "SE_NOME_NIVELi"
                nome_bus = f"{nome_se}_NIVEL{i + 1}"

                if nome_bus not in network.buses.index:
                    network.add("Bus",
                                name=nome_bus,
                                v_nom=tensao,
                                x=sub.geometry.x,
                                y=sub.geometry.y)

                # Adicionar transformadores entre os níveis (se não for o primeiro)
                if i > 0:
                    nome_trafo = f"Trafo_{nome_se}_{i}-{i + 1}"

                    if nome_trafo not in network.transformers.index:
                        # Parâmetros típicos de transformador
                        snom = 100  # MVA
                        perdas = 0.005  # 0.5%

                        network.add("Transformer",
                                    name=nome_trafo,
                                    bus0=f"{nome_se}_NIVEL{i}",
                                    bus1=f"{nome_se}_NIVEL{i + 1}",
                                    s_nom=snom,
                                    x=0.1,  # Reatância
                                    r=perdas,  # Resistência
                                    tap_ratio=1.0,
                                    phase_shift=0)

    # Verificar barramentos criados (corrigido)
    print(f"\nResumo da rede:")
    print(f"- Total de barramentos: {len(network.buses)}")
    print(f"- Total de transformadores: {len(network.transformers)}")

    # Listar subestações com múltiplas tensões
    print("\nSubestações com múltiplas tensões:")
    for bus in network.buses.index[network.buses.index.str.contains('_NIVEL')]:
        v_nom = network.buses.at[bus, 'v_nom']
        print(f"- {bus} ({v_nom / 1e3} kV)")


def adicionar_geradores_completo(network, geradores_gdf, linhas_conectadas, subestacoes_gdf):
    """
    Adiciona geradores à rede PyPSA considerando:
    - Subestações com múltiplos níveis de tensão
    - Conexão de geradores no nível mais baixo
    - Verificação completa de inserção

    Retorna:
        - network: Rede PyPSA modificada
        - report: Dicionário com detalhes de inserção
    """
    # 1. Configuração inicial
    CRS_PROJETADO = "EPSG:31983"  # UTM 23S para Brasil
    geradores_gdf = geradores_gdf.to_crs(CRS_PROJETADO)
    linhas_conectadas = linhas_conectadas.to_crs(CRS_PROJETADO)
    subestacoes_gdf = subestacoes_gdf.to_crs(CRS_PROJETADO)

    report = {
        'total_geradores': len(geradores_gdf),
        'inseridos': set(),
        'nao_inseridos': set(),
        'erros': {},
        'conexoes_por_nivel': {}
    }

    # 2. Pré-processamento de subestações multi-nível
    def encontrar_nivel_mais_baixo(se_base):
        """Encontra o barramento com menor tensão para uma SE"""
        niveis = [bus for bus in network.buses.index
                  if bus.startswith(f"{se_base}_NIVEL")]

        if not niveis:
            return se_base  # SE sem múltiplos níveis

        # Encontrar nível com menor tensão
        niveis_com_tensao = [
            (bus, network.buses.at[bus, 'v_nom'])
            for bus in niveis
        ]
        return min(niveis_com_tensao, key=lambda x: x[1])[0]

    # 3. Identificar geradores origem
    geradores_origem = {
        str(linha['SE_origem']).strip()
        for _, linha in linhas_conectadas.iterrows()
        if isinstance(linha['SE_origem'], str)
           and not linha['SE_origem'].startswith('SE ')
    }

    # 4. Processar cada gerador
    for _, gerador in geradores_gdf.iterrows():
        nome_ger = str(gerador['nome']).strip()
        try:
            # Encontrar SE mais próxima
            pt_gerador = Point(gerador.geometry.x, gerador.geometry.y)
            distancias = subestacoes_gdf.geometry.distance(pt_gerador)
            se_proxima = subestacoes_gdf.iloc[distancias.idxmin()]
            nome_se_base = str(se_proxima['Nome']).strip()

            # Determinar barramento de conexão (nível mais baixo)
            barramento_conexao = encontrar_nivel_mais_baixo(nome_se_base)

            if barramento_conexao not in network.buses.index:
                report['nao_inseridos'].add(nome_ger)
                report['erros'][nome_ger] = f"Barramento {barramento_conexao} não encontrado"
                continue

            tensao = network.buses.at[barramento_conexao, 'v_nom']
            potencia = float(gerador['potencia']) * 1e6  # MW → W

            # Registrar nível de tensão usado
            nivel = barramento_conexao.split('_NIVEL')[-1] if '_NIVEL' in barramento_conexao else '0'
            report['conexoes_por_nivel'][nome_ger] = {
                'barramento': barramento_conexao,
                'tensao_kV': tensao / 1e3,
                'nivel': nivel
            }

            # Caso 1: Gerador é origem de linha
            if nome_ger in geradores_origem:
                nome_bus = f"BUS_{nome_ger.replace(' ', '_')}"

                if nome_bus not in network.buses.index:
                    network.add("Bus",
                                name=nome_bus,
                                v_nom=tensao,
                                x=gerador.geometry.x,
                                y=gerador.geometry.y)

                network.add("Generator",
                            name=nome_ger,
                            bus=nome_bus,
                            p_nom=potencia,
                            **{k: v for k, v in gerador.items()
                               if k not in ['geometry', 'nome'] and pd.notna(v)})

                network.add("Line",
                            name=f"LIG_{nome_ger.replace(' ', '_')}",
                            bus0=nome_bus,
                            bus1=barramento_conexao,
                            length=0.1,
                            type="NA2XS2Y 1x240 RM/25 12.0")

            # Caso 2: Gerador normal
            else:
                network.add("Generator",
                            name=nome_ger,
                            bus=barramento_conexao,
                            p_nom=potencia,
                            **{k: v for k, v in gerador.items()
                               if k not in ['geometry', 'nome'] and pd.notna(v)})

            report['inseridos'].add(nome_ger)

        except Exception as e:
            report['nao_inseridos'].add(nome_ger)
            report['erros'][nome_ger] = str(e)
            continue

    # 5. Verificação final e relatório
    geradores_na_rede = set(network.generators.index)
    for nome_ger in geradores_gdf['nome'].str.strip():
        if nome_ger not in geradores_na_rede and nome_ger not in report['nao_inseridos']:
            report['nao_inseridos'].add(nome_ger)
            report['erros'][nome_ger] = "Não inserido (razão desconhecida)"

    print("\n" + "=" * 50)
    print("RELATÓRIO DE INSERÇÃO DE GERADORES")
    print(f"Total processado: {report['total_geradores']}")
    print(f"Inseridos com sucesso: {len(report['inseridos'])}")
    print(f"Não inseridos: {len(report['nao_inseridos'])}")

    if report['nao_inseridos']:
        print("\nDetalhes dos geradores não inseridos:")
        for gerador in sorted(report['nao_inseridos']):
            print(f"- {gerador}: {report['erros'].get(gerador, 'Erro não especificado')}")

    print("\nDistribuição por níveis de tensão:")
    dist_niveis = pd.DataFrame.from_dict(report['conexoes_por_nivel'], orient='index')
    print(dist_niveis['nivel'].value_counts().sort_index())

    print("=" * 50 + "\n")

    return network, report

def verificar_conexoes_se(linhas_conectadas, subestacoes_gdf):
    """
    Verifica se todas as SEs de origem e destino das linhas existem no DataFrame de subestações.

    Args:
        linhas_conectadas: GeoDataFrame com as linhas de transmissão conectadas
        subestacoes_gdf: GeoDataFrame com as subestações

    Returns:
        Tuple: (bool indicando sucesso, DataFrame com linhas problemáticas)
    """
    # 1. Extrair nomes únicos de subestações válidas
    se_validas = set(subestacoes_gdf['Nome'].unique())

    # 2. Verificar cada linha
    problemas = []
    for idx, linha in linhas_conectadas.iterrows():
        origem_ok = pd.isna(linha['SE_origem']) or (linha['SE_origem'] in se_validas)
        destino_ok = pd.isna(linha['SE_destino']) or (linha['SE_destino'] in se_validas)

        if not (origem_ok and destino_ok):
            problemas.append({
                'id_linha': idx,
                'SE_origem': linha['SE_origem'],
                'SE_destino': linha['SE_destino'],
                'problema': 'SE_origem não encontrada' if not origem_ok else 'SE_destino não encontrada'
            })

    # 3. Criar DataFrame de problemas
    df_problemas = pd.DataFrame(problemas)

    # 4. Retornar resultado
    if len(df_problemas) > 0:
        print(f"\nAviso: {len(df_problemas)} linhas com problemas de conexão:")
        print(df_problemas[['id_linha', 'SE_origem', 'SE_destino', 'problema']])
        return False, df_problemas
    else:
        print("Todas as conexões estão válidas!")
        return True, None

CRS_PROJETADO = "EPSG:31983"  # UTM 23S para Brasil
linhas = carregar_LTs()
linhas = linhas.set_crs(CRS_PROJETADO)
subestacoes = carregar_SEs()
subestacoes = subestacoes.set_crs(CRS_PROJETADO)
geradores = carregar_Geracao()
geradores = geradores.set_crs(CRS_PROJETADO)




# Visualizar o resultado

network = pypsa.Network()

adicionar_subestacoes(network, subestacoes)

linhas_conectadas = conectar_linhas_subestacoes(linhas, subestacoes, geradores)
adicionar_geradores_completo(network, geradores, linhas_conectadas, subestacoes)
# Visualizar resultados
print(network.buses)