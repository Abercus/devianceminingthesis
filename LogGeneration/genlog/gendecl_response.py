

def activity(name):
    return "activity {}\n".format(name)

def existence(name):
    return "Existence[{}] | |\n".format(name)

def response(name1, name2):
    return "Response[{}, {}] | |\n".format(name1, name2)

def not_response(name1, name2):
    return "NotResponse[{}, {}] | |\n".format(name1, name2)

def not_chain_response(name1, name2):
    return "NotChainResponse[{}, {}] | |\n".format(name1, name2)

def absence(name):
    return "Absence[{}] | |\n".format(name)

def absence_c(name, c):
    return "Absence[{},{}] | |\n".format(name, c)

def generate_random_activities(n=100):
    activity_names = []
    for i in range(1, n+1):
        activity_names.append("random_{}".format(i))

    return activity_names


def gen_deviant_1():
    
    t_names = ["response_A", "response_B"]
    r_names = generate_random_activities(100)

    activity_names = t_names + r_names
    with open("dev1.decl", "w") as f:
        for name in activity_names:
            f.write(activity(name))
        f.write("\n")
        # write constraitns
        f.write(response("response_A", "response_B"))
        f.write(existence("response_A"))
        
        # write absences
        for name in activity_names:
            f.write(absence_c(name, 4))

def gen_deviant_2():
    t_names = ["response_A", "response_B"]
    r_names = generate_random_activities(100)
    
    activity_names = t_names + r_names
    with open("dev2.decl", "w") as f:
        for name in activity_names:
            f.write(activity(name))
        f.write("\n")
        # write constraitns
        f.write(response("response_A", "response_B"))
        f.write(not_chain_response("response_A", "response_B"))
        f.write(existence("response_A"))
        
        # write absences
        for name in activity_names:
            f.write(absence_c(name, 4))


def gen_norm_1():
    t_names = ["response_A", "response_B"]
    r_names = generate_random_activities(100)
    
    activity_names = t_names + r_names
    with open("norm1.decl", "w") as f:
        for name in activity_names:
            f.write(activity(name))
        f.write("\n")
        # write constraitns
        f.write(not_response("response_A", "response_B"))
        f.write(existence("response_A"))
        
        # write absences
        for name in activity_names:
            f.write(absence_c(name, 4))

def gen_norm_2():
    t_names = ["response_A", "response_B"]
    r_names = generate_random_activities(100)
    
    activity_names = t_names + r_names
    with open("norm2.decl", "w") as f:
        for name in activity_names:
            f.write(activity(name))
        f.write("\n")
        # write constraints
        f.write(not_response("response_A", "response_B"))
        f.write(existence("response_B"))
        
        # write absences
        for name in activity_names:
            f.write(absence_c(name, 4))

def gen_norm_3():
    t_names = ["response_A", "response_B"]
    r_names = generate_random_activities(100)
    
    activity_names = t_names + r_names
    with open("norm3.decl", "w") as f:
        for name in activity_names:
            f.write(activity(name))
        f.write("\n")
        # write constraints
        f.write(not_response("response_A", "response_B"))
        f.write(existence("response_A"))
        f.write(existence("response_B"))

        # write absences
        for name in r_names:
            f.write(activity_names(name, 4))


def gen_norm_4():
    t_names = ["response_A", "response_B"]
    r_names = generate_random_activities(100)
    
    activity_names = t_names + r_names
    with open("norm4.decl", "w") as f:
        for name in activity_names:
            f.write(activity(name))
        f.write("\n")
        # write constraitns
        f.write(absence("response_A"))
        f.write(absence("response_B"))

        # write absences
        for name in activity_names:
            f.write(absence_c(name, 4))






if __name__ == "__main__":
    gen_deviant_1()
    gen_deviant_2()
    gen_norm_1()
    gen_norm_2()
    gen_norm_3()
    gen_norm_4()
