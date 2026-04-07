def allocate_resources(tb_cases):

    beds = int(tb_cases * 0.25)
    doctors = int(tb_cases / 100)
    test_kits = int(tb_cases * 1.2)

    return beds, doctors, test_kits