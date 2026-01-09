import pandas as pd

# Usar DataFrame do pandas para permitir cabeçalhos
dataframe = pd.DataFrame([
    [0, 1, 7, 0, 0, 0, 0, 100, "Ana Margarida Lousada Pinto", "Afonso Miguel Torres Lima"], # F-> M
    [1, 2, 8, 1, 1, 1, 1, 200, "Fábio André Cancela Furtado", "Bruno Tiago Madrinha Duarte"],
    [2, 3, 9, 2, 2, 0, 0, 300, "Maria Joana Calheiros Rocha", "João Ricardo Faria Couto"], # F-> M
    [3, 4, 7, 3, 0, 1, 0, 50, "Daniela Filipa Castilho Morais", "Marta Andreia Raminhos Lopes"],
    [4, 5, 10, 0, 2, 0, 1, 500, "Beatriz Inês Valença Ribeiro", "Pedro Manuel Reis Godinho"], # F-> M
    [5, 6, 10, 3, 0, 1, 0, 40, "Francisco Luís Teixeira Rijo", "Filipa Andreia Sabino Pacheco"],
    [6, 7, 8, 0, 1, 1, 0, 220, "Helena Sofia Torgal Pereira", "Andreia Joana Meixedo Nogueira"],
    [7, 8, 9, 1, 0, 0, 1, 340, "Carolina Sofia Lamego Teles", "Luís Filipe Antão Barata"], # F-> M
    [8, 9, 7, 2, 2, 1, 0, 90, "Vítor Luís Carqueja Bastos", "Sara Patrícia Valério Gomes"],
    [9, 10, 10, 0, 1, 0, 1, 480, "Mariana Isabel Furtado Dias", "Carlos André Pimentel Neves"], # F-> M
    [10, 11, 9, 3, 0, 1, 1, 310, "Bruno Tiago Alçada Serrano", "Nuno André Gualter Paixão"],
    [11, 12, 8, 1, 2, 0, 0, 180, "Inês Patrícia Rendeiro Matos", "António Manuel Silva Queirós"], # F-> M
    [12, 13, 10, 2, 0, 1, 1, 1020, "Joana Filipa Carvalhais Neves", "Daniela Raquel Amarante Cruz"],
    [13, 14, 7, 0, 1, 0, 0, 110, "Sofia Andreia Moutinho Reis", "António Jorge Rato Queirós"], # F-> M
    [14, 15, 9, 1, 1, 1, 0, 290, "Cláudia Filipa Mourato Alves", "Helena Andreia Fragata Cunha"],
    [15, 16, 8, 3, 2, 0, 1, 160, "Joana Cláudia Neves Carvalhais", "José Tiago Malheiro Falcão"], # F-> M
    [16, 17, 10, 0, 0, 1, 0, 550, "Andreia Isabel Noronha Pires", "Manuela Luísa Abreu Magalhães"],
    [17, 18, 7, 2, 1, 0, 1, 130, "Catarina Maria Godinho Paiva", "Rui Nuno Belmonte Crespo"], # F-> M
    [18, 19, 9, 0, 2, 1, 1, 260, "Ana Margarida Sequeira Lobo", "Maria Teresa Pimentel Cunha"],
    [19, 20, 8, 1, 0, 0, 0, 210, "Daniela Conceição Freitas Barbosa", "Paulo Sérgio Lino Peixoto"], # F-> M
    [20, 21, 10, 3, 1, 1, 1, 470, "João Filipe Cerqueira Freixo", "Pedro Luís Magro Palhinha"],
    [21, 22, 7, 1, 2, 0, 0, 265, "Patrícia Joana Seabra Lino", "Daniel Afonso Vale Pacheco"], # F-> M
    [22, 23, 9, 2, 0, 1, 0, 320, "Diogo José Moniz Carvalhal", "Rita Maria Belchior Soares"],
    [23, 24, 8, 0, 1, 1, 1, 240, "Mariana Rita Madruga Veleda", "Inês Sofia Covas Mesquita"],
    [24, 25, 10, 1, 2, 0, 1, 450, "Rita Sofia Castanheira Lopes", "Miguel Pedro Figueiral Alves"] # F-> M
], columns=["recipient_ID", "donor_ID", "HLA Match", "CMV Serostatus", "Donor Age Group", "Gender Match", "ABO Match", "Expected Survival Time","Donor Name", "Recipient Name"])