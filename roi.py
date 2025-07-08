def calcular_roi(custo_total, ganho_total):
    """
    Calcula o Retorno sobre o Investimento (ROI).

    Args:
        custo_total (float): O valor total do investimento.
        ganho_total (float): O valor total dos ganhos gerados pelo investimento.

    Returns:
        float: O ROI em porcentagem. Retorna None se o custo_total for zero para evitar divisão por zero.
    """
    if custo_total == 0:
        return None 

    roi = ((ganho_total - custo_total) / custo_total) * 100
    return roi

if __name__ == "__main__":
    print("--- Calculadora de ROI para Projetos de ML ---")
    print("\nInsira os valores estimados:")


   
    print("\n--- Custos do Projeto ---")
    print("Considere: Tempo de desenvolvimento, hardware/software dedicados, energia, etc.")

    custo_desenvolvimento_horas = 250
    valor_hora_desenvolvedor = 40.0
    custo_total_hardware = 200.0 
    custo_total_software_nuvem = 150.0 
    custo_total_outros = 50.0 


    print("\n--- Ganhos do Projeto ---")
    print("Considere: Aprendizado (aumento do valor profissional), economia de tempo, base para produto futuro, etc.")

 
    ganho_valor_profissional = 15000.0
    ganho_melhoria_pratica_musical = 2500.0 
    ganho_potencial_produto = 5000.0 
    ganho_direto_vendas = 0.0 


    custo_total = (custo_desenvolvimento_horas * valor_hora_desenvolvedor) + \
                  custo_total_hardware + \
                  custo_total_software_nuvem + \
                  custo_total_outros

    ganho_total = ganho_valor_profissional + \
                  ganho_melhoria_pratica_musical + \
                  ganho_potencial_produto + \
                  ganho_direto_vendas

    print(f"\nCusto Total Estimado: R$ {custo_total:,.2f}")
    print(f"Ganho Total Estimado: R$ {ganho_total:,.2f}")

    roi_calculado = calcular_roi(custo_total, ganho_total)

    if roi_calculado is not None:
        print(f"\nROI: {roi_calculado:,.2f}%")
        if roi_calculado > 0:
            print("Isso indica que o projeto gerou um retorno positivo sobre o investimento.")
        elif roi_calculado < 0:
            print("Isso indica que o projeto gerou um retorno negativo sobre o investimento.")
        else:
            print("O projeto empatou (ganhos iguais aos custos).")
    else:
        print("\nNão foi possível calcular o ROI, pois o custo total é zero.")

    print("\n--- Fim da Calculadora ---")
    print("Lembre-se de ajustar os valores com base na realidade do seu projeto!")