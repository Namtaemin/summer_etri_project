# Unity

1. DropDownExample.cs : 드롭다운 버튼을 클릭했을 때의 이벤트 처리
3. FuzzyExam.cs : 온도,습도,조도의 값을 간략히 표현한 데이터 값을 서버로부터 받아옴
4. JsonExample.cs : 현재온도, 현재습도, 현재조도값을 서버로부터 받아고, 서버로 값을 보냄
6. buttonEvent,1,2.cs : 냉난방기, 제습기, 조명을 눌렀을 때의 이벤트 파티클시스템이 작동함
7. tempinput.cs : inputField에 원하는 수를 적어, 현재온도, 현재습도, 현재조도 Text 값을 변경해줌
8. fuzzyy.py : 퍼지이론 대신하여 만든 if문으로 가중치값을 결정하여 반환해줌

# DropDownExample.cs

* DropDown Button을 활용하기 위해서 Dropdown 객체하나를 생성하고, 온도, 습도, 조도를 텍스트에 나타낼 수 있게 3개의 Text 객체를 선언한다. POST 통신을 할 수 있게 어떤 URL을 이용할 것인지 BaseURL을 지정하고, Json 데이터를 받아올 수 있게 데이터 직렬화를 하고, GetData() 함수를 선언한다.


* DropDown의 목록을 생성한다. 1시간부터 12시간 이후까지의 데이터값을 받아올 것이기 때문에 12개의 항목을 생성한다. 이렇게 되면, 각 하나의 항목을 지정할 때마다 한시간 이후, 두시간 이후 같이 예측한 데이터값이 텍스트에 출력될 수 있도록 해야한다. 따라서 SelectButton이란 함수를 생성하여 클릭될 때마다 서버로부터 값을 받아와 Text에 데이터값을 출력할 수 있도록한다.


* SelectButton에서 iter라는 변수로 몇시간 이후의 데이터를 받아올지 정하고, case문을 이용하여 0~11까지의 이 값을 서버로 보내면, 이 값에 해당하는 예측 데이터를 받아온다. 받아온 값을 Text에 출력하면 된다.

# ButtonEvent, 1, 2
* Particle System을 on, off를 할 수 있게 해주는 버튼이다. 버튼을 누르게되면 파티클 시스템이 켜지고, 다시 누르게 되면 파티클 시스템 작동이 멈추게 된다.
* 미구현 : 버튼을 누르게 되면 퍼지값을 출력하는 텍스트 변화를 아직 구현안함.

# FutureExample
* DropDownExample의 한 부분이 되어버린 코드이다. getRandom API에서 데이터 값을 받아와 텍스트에 출력하려고 했던 test용 코드이다.

# FuzzyExam
* 퍼지의 값을 서버로부터 받아와 해당 Text에 출력하려고했던 cs 파일이다. 
* 현 상황엔 아직 구현이 되지 않았고, tempinput 파일에 이 코드도 들어가서 퍼지값까지 한번에 받아올 것 같다.

#JsonExample
* 현재 온,습,조도의 데이터 값을 서버로부터 받아온다. 

#tempinput
* 사용자가 원하는 온도, 습도, 조도 값을 입력하고, 조절 버튼을 누르게 되면 현재 온도, 습도, 조도의 값이 변하게 된다.
