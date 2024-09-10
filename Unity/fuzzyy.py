# 습하다 = 1.0
# 어둡다 = 1.0 
# 덥다 = 1.0

def VirtualSpace(temperature, humidity, illuminance, input_temp, input_humi, et, eh):
    temp_weight = 0
    humi_weight = 0
    illum_weight = 0
    real_humi_weight = 0   

    if((temperature >= input_temp - et) and (temperature <= input_temp + et)):
        temp_weight = 0.5
    elif((temperature >= input_temp + et) and (temperature <= input_temp + et*2)):
        temp_weight = 0.7
    elif(temperature >= input_temp + et*2):
        temp_weight = 1.0
    elif((temperature >= input_temp - et*2) and (temperature <= input_temp - et)):
        temp_weight = 0.3
    elif((temperature >= input_temp - et*3) and (temperature <= input_temp - et*2)):
        temp_weight = 0.1

    if((humidity >= input_humi - eh) and (humidity <= input_humi + eh)):
        humi_weight = 0.5
    if((humidity >= input_humi + eh) and (humidity <= input_humi + eh*2)):
        humi_weight = 0.7
    if(humidity >= input_humi + eh*2):
        humi_weight = 1.0
    if((humidity >= input_humi - eh*2) and (humidity <= input_humi - eh)):
        humi_weight = 0.3
    if((humidity >= input_humi - eh*3) and (humidity <= input_humi - eh*2)):
        humi_weight = 0.1

    if(illuminance >= 21000):
        illum_weight = 1.0
    elif(illuminance < 21000):
        illum_weight = 0.1
    
    if(temp_weight == 1.0):
        if(humi_weight == 1.0):
            real_humi_weight = 1.0
        elif(humi_weight == 0.7):
            real_humi_weight = 0.7
        elif(humi_weight == 0.5):
            real_humi_weight = 0.5
        elif(humi_weight == 0.3):
            real_humi_weight = 0.3
        elif(humi_weight == 0.1):
            real_humi_weight = 0.1

    elif(temp_weight == 0.7):
        if(humi_weight == 1.0):
            real_humi_weight = 1.0
        elif(humi_weight == 0.7):
            real_humi_weight = 0.7
        elif(humi_weight == 0.5):
            real_humi_weight = 0.5
        elif(humi_weight == 0.3):
            real_humi_weight = 0.3
        elif(humi_weight == 0.1):
            real_humi_weight = 0.1

    elif(temp_weight == 0.5):
        if(humi_weight == 1.0):
            real_humi_weight = 0.7
        elif(humi_weight == 0.7):
            real_humi_weight = 0.5
        elif(humi_weight == 0.5):
            real_humi_weight = 0.5
        elif(humi_weight == 0.3):
            real_humi_weight = 0.3    
        elif(humi_weight == 0.1):
            real_humi_weight = 0.1

    elif(temp_weight == 0.3):
        if(humi_weight == 1.0):
            real_humi_weight = 0.7
        elif(humi_weight == 0.7):
            real_humi_weight = 0.5
        elif(humi_weight == 0.5):
            real_humi_weight = 0.3
        elif(humi_weight == 0.3):
            real_humi_weight = 0.3
        elif(humi_weight == 0.1):
            real_humi_weight = 0.1

    elif(temp_weight == 0.1):
        if(humi_weight == 1.0):
            real_humi_weight = 0.7
        elif(humi_weight == 0.7):
            real_humi_weight = 0.7
        elif(humi_weight == 0.5):
            real_humi_weight = 0.3
        elif(humi_weight == 0.3):
            real_humi_weight = 0.3
        elif(humi_weight == 0.1):
            real_humi_weight = 0.1
    
    return temp_weight, real_humi_weight, illum_weight

temperature = 30
humidity= 70
illuminance = 22000

print(VirtualSpace(temperature, humidity, illuminance, 21, 40, 1.5, 10))