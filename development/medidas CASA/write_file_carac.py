def write_file_carac(CARAC_WHO):
    
    with open('features_who_test.csv', 'w') as file:
        for carac in CARAC_WHO:
            for i in range(len(carac)):
                file.write(str(carac[i]))
                if i != len(carac) - 1:
                    file.write(',')
            file.write('\n')

        file.write('\n')

    file.close()
    return