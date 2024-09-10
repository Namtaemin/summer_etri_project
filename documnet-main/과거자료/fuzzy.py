temperature = float(input("온도를 입력하세요 : "))
humidity = float(input("습도를 입력하세요 : "))
illuminance = float(input("조도를 입력하세요 : "))

if(temperature >= 26):
    print("덥습니다.")
    if(temperature >= 24):
        if(humidity >= 32 and humidity <= 38):
            print("적당한 습도입니다.")
            if(illuminance >= 24000):
                print("어둡습니다.")
            elif(illuminance < 24000 and illuminance >= 21000):
                print("적당한 밝기입니다.")
            elif(illuminance < 21000):
                print("밝습니다.")
        elif(humidity > 39 and humidity <= 60):
            print("약간 습합니다.")
            if(illuminance >= 24000):
                print("어둡습니다.")
            elif(illuminance < 24000 and illuminance >= 21000):
                print("적당한 밝기입니다.")
            elif(illuminance < 21000):
                print("밝습니다.")
        elif(humidity >60):
            print("습합니다.")
            if(illuminance >= 24000):
                print("어둡습니다.")
            elif(illuminance < 24000 and illuminance >= 21000):
                print("적당한 밝기입니다.")
            elif(illuminance < 21000):
                print("밝습니다.")
        elif(humidity < 32):
            print("건조합니다.")
            if(illuminance >= 24000):
                print("어둡습니다.")
            elif(illuminance < 24000 and illuminance >= 21000):
                print("적당한 밝기입니다.")
            elif(illuminance < 21000):
                print("밝습니다.")


elif(temperature < 26 and temperature >= 20):
    print("적당한 온도입니다.")
    if(temperature <= 23 and temperature >= 21):
        if(humidity >= 37 and humidity <= 43):
            print("적당한 습도입니다.")
            if(illuminance >= 24000):
                print("어둡습니다.")
            elif(illuminance < 24000 and illuminance >= 21000):
                print("적당한 밝기입니다.")
            elif(illuminance < 21000):
                print("밝습니다.")
        elif(humidity > 43 and humidity <= 60):
            print("약간 습합니다.")
            if(illuminance >= 24000):
                print("어둡습니다.")
            elif(illuminance < 24000 and illuminance >= 21000):
                print("적당한 밝기입니다.")
            elif(illuminance < 21000):
                print("밝습니다.")
        elif(humidity > 60):
            print("습합니다.")
            if(illuminance >= 24000):
                print("어둡습니다.")
            elif(illuminance < 24000 and illuminance >= 21000):
                print("적당한 밝기입니다.")
            elif(illuminance < 21000):
                print("밝습니다.")
        elif(humidity < 37 and humidity >= 30):
            print("약간 건조합니다.")
            if(illuminance >= 24000):
                print("어둡습니다.")
            elif(illuminance < 24000 and illuminance >= 21000):
                print("적당한 밝기입니다.")
            elif(illuminance < 21000):
                print("밝습니다.")        
        elif(humidity < 30 ):
            print("건조합니다.")
            if(illuminance >= 24000):
                print("어둡습니다.")
            elif(illuminance < 24000 and illuminance >= 21000):
                print("적당한 밝기입니다.")
            elif(illuminance < 21000):
                print("밝습니다.")
        
elif(temperature <20 and temperature >= 15.6):
    print("적정 온도입니다.")
    if(temperature >= 18 and temperature <= 20):
        if(humidity >= 47 and humidity <= 53):
            print("적정 습도입니다.")
            if(illuminance >= 24000):
                print("어둡습니다.")
            elif(illuminance < 24000 and illuminance >= 21000):
                print("적당한 밝기입니다.")
            elif(illuminance < 21000):
                print("밝습니다.")
        elif(humidity > 53):
            print("습합니다.")
            if(illuminance >= 24000):
                print("어둡습니다.")
            elif(illuminance < 24000 and illuminance >= 21000):
                print("적당한 밝기입니다.")
            elif(illuminance < 21000):
                print("밝습니다.")
        elif(humidity < 47 and humidity >=35):
            print("약간 건조합니다.")
            if(illuminance >= 24000):
                print("어둡습니다.")
            elif(illuminance < 24000 and illuminance >= 21000):
                print("적당한 밝기입니다.")
            elif(illuminance < 21000):
                print("밝습니다.")
        elif(humidity < 35):
            print("건조합니다.")
            if(illuminance >= 24000):
                print("어둡습니다.")
            elif(illuminance < 24000 and illuminance >= 21000):
                print("적당한 밝기입니다.")
            elif(illuminance < 21000):
                print("밝습니다.")
        
elif(temperature <15.6):
    print("추운 온도입니다.")
    if(humidity>=57 and humidity <= 63):
        print("적정 습도입니다.")
        if(illuminance >= 24000):
                print("어둡습니다.")
        elif(illuminance < 24000 and illuminance >= 21000):
                print("적당한 밝기입니다.")
        elif(illuminance < 21000):
                print("밝습니다.")
    elif(humidity > 63 ):
        print("습합니다.")
        if(illuminance >= 24000):
                print("어둡습니다.")
        elif(illuminance < 24000 and illuminance >= 21000):
                print("적당한 밝기입니다.")
        elif(illuminance < 21000):
                print("밝습니다.")
        
    elif(humidity < 57 and humidity >= 40):
        print("약간 건조합니다.")
        if(illuminance >= 24000):
                print("어둡습니다.")
        elif(illuminance < 24000 and illuminance >= 21000):
                print("적당한 밝기입니다.")
        elif(illuminance < 21000):
                print("밝습니다.")
    elif(humidity < 40):
        print("건조합니다.")
        if(illuminance >= 24000):
                print("어둡습니다.")
        elif(illuminance < 24000 and illuminance >= 21000):
                print("적당한 밝기입니다.")
        elif(illuminance < 21000):
                print("밝습니다.")

# if(temperature >= 26):
#     print("덥습니다.")
#     if(temperature >= 24):
#         if(humidity >= 32 and humidity <= 38):
#             print("적당한 습도입니다.")
#         elif(humidity > 39 and humidity <= 60):
#             print("약간 습합니다.")
#         elif(humidity >60):
#             print("습합니다.")
#         elif(humidity < 32):
#             print("건조합니다.")
            

# elif(temperature < 26 and temperature >= 20):
#     print("적당한 온도입니다.")
#     if(temperature <= 23 and temperature >= 21):
#         if(humidity >= 37 and humidity <= 43):
#             print("적당한 습도입니다.")
#         elif(humidity > 43 and humidity <= 60):
#             print("약간 습합니다.")
#         elif(humidity > 60):
#             print("습합니다.")
#         elif(humidity < 37 and humidity >= 30):
#             print("약간 건조합니다.")        
#         elif(humidity < 30 ):
#             print("건조합니다.")
        
# elif(temperature <20 and temperature >= 15.6):
#     print("적정 온도입니다.")
#     if(temperature >= 18 and temperature <= 20):
#         if(humidity >= 47 and humidity <= 53):
#             print("적정 습도입니다.")
#         elif(humidity > 53):
#             print("습합니다.")
#         elif(humidity < 47 and humidity >=35):
#             print("약간 건조합니다.")
#         elif(humidity < 35):
#             print("건조합니다.")
        
# elif(temperature <15.6):
#     print("추운 온도입니다.")
#     if(humidity>=57 and humidity <= 63):
#         print("적정 습도입니다.")
#     elif(humidity > 63 ):
#         print("습합니다.")
#     elif(humidity < 57 and humidity >= 40):
#         print("약간 건조합니다.")
#     elif(humidity < 40):
#         print("건조합니다.")



# # 습도 자체를 온도 아래에 넣어버린다. 
# # 조도 또한 넣어버려서 if문이 끝나니..


