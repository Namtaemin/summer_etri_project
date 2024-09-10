using System.Collections;
using System.Collections.Generic;
using System.Net;
using System.IO;
using UnityEngine;
using UnityEngine.Networking;
using UnityEngine.UI;

 // Required when Using UI elements.
// temp, humi, illu 텍스트를 서버에 보내면, 서버가 받은 후 퍼지로 변경해줘서 바뀌는 형식.
// [System.Serializable]
// public class LoginFuzzyData
// {
// }

// [System.Serializable]
// public class FuzzyData{
//     public string controlT;
//     public string controlH;
//     public string controlI;
//     public float stateT;
//     public float stateH;
//     public float stateI;
    
// }

[System.Serializable]
public class Filed
{
    public float temperature;
    public float humidity; 
    public float illuminance;
    public float errorT;
    public float errorH;
}

public class tempinput : MonoBehaviour 
{
    public Text temp;
    public Text humi;
    public Text illu; 
    public Text tempErrorValue;
    public Text humiErrorValue;
    public InputField inputField;
    public InputField inputField2; 
    public InputField inputField3;
    public InputField inputField4;
    public InputField inputField5;  
    public Button button;
    // public Text TemperatureText;
    // public Text HumidityText;
    // public Text IlluminanceText;
    
    void Start()
    {
        StartCoroutine(SetFunction_UI());
    }

    IEnumerator SetFunction_UI()
    {
        //Reset
        ResetFunction_UI();

        button.onClick.AddListener(Function_Button);
        inputField.onValueChanged.AddListener(Function_InputField); 
        inputField.onEndEdit.AddListener(Function_InputField_EndEdit); 
        inputField2.onValueChanged.AddListener(Function_InputField); 
        inputField2.onEndEdit.AddListener(Function_InputField_EndEdit); 
        inputField3.onValueChanged.AddListener(Function_InputField); 
        inputField3.onEndEdit.AddListener(Function_InputField_EndEdit);
        inputField4.onValueChanged.AddListener(Function_InputField); 
        inputField4.onEndEdit.AddListener(Function_InputField_EndEdit);
        inputField5.onValueChanged.AddListener(Function_InputField); 
        inputField5.onEndEdit.AddListener(Function_InputField_EndEdit);                 
        yield return null;
    }

    IEnumerator Upload(string URL,string json)
    {
        using (UnityWebRequest request = UnityWebRequest.Post(URL, json))
        {
            byte[] jsonToSend = new System.Text.UTF8Encoding().GetBytes(json);
            request.uploadHandler = new UploadHandlerRaw(jsonToSend);
            request.downloadHandler = (DownloadHandler)new DownloadHandlerBuffer();
            request.SetRequestHeader("Content-Type", "application/json");

            yield return request.SendWebRequest();
            Debug.Log(request.downloadHandler.text);

            Filed fileddata = new Filed();
            
            fileddata = JsonUtility.FromJson<Filed>(request.downloadHandler.text);
            string fileName = "Myoutputjson";
            string path = Application.dataPath + "/Resource/Json/"+ fileName + ".Json";

            File.WriteAllText(path, request.downloadHandler.text);
            
            string filePath = "Assets/resource/Json/Myoutputjson.json";
            string jsonon =File.ReadAllText(filePath);

            Debug.Log(fileddata.temperature);
            Debug.Log(fileddata.humidity);
            Debug.Log(fileddata.illuminance);            
            inputField.text = fileddata.temperature.ToString();
            inputField2.text = fileddata.humidity.ToString();
            inputField3.text = fileddata.illuminance.ToString();
            inputField4.text = fileddata.errorT.ToString();
            inputField5.text = fileddata.errorH.ToString();

        }
    }
// 퍼지 온습도조도를 받아와야함. temperature.text에서 받는것이 아니다.
    // private FuzzyData GetMemberData()
    // {
    //     LoginFuzzyData data3 = new LoginFuzzyData();

    //     string str3 = JsonUtility.ToJson(data3);
    //     var bytes = System.Text.Encoding.UTF8.GetBytes(str3);

    //     HttpWebRequest request = (HttpWebRequest)WebRequest.Create("http://192.168.0.2:5000/getControl");
    //     request.Method = "POST";
    //     request.ContentType = "application/json";
    //     request.ContentLength = bytes.Length;

    //     using(var stream = request.GetRequestStream())
    //     {
    //         stream.Write(bytes, 0, bytes.Length);
    //         stream.Flush();
    //         stream.Close();
    //     }

    //     HttpWebResponse response = (HttpWebResponse)request.GetResponse();
    //     StreamReader reader = new StreamReader(response.GetResponseStream());
    //     string json3 = reader.ReadToEnd();
    //     FuzzyData info = JsonUtility.FromJson<FuzzyData>(json3);

    //     Debug.Log("temperature : "+info.stateT);
    //     Debug.Log("humidity : "+info.stateH);
    //     Debug.Log("illuminance : "+info.stateI);

    //     // TemperatureText.text= info.controlT.ToString();
    //     // HumidityText.text= info.controlH.ToString();
    //     // IlluminanceText.text= info.controlI.ToString();
        
    //     return info;
    // }

    private void Function_Button()
    {
        string txt = inputField.text;
        string txt2 = inputField2.text;
        string txt3 = inputField3.text;
        string txt4 = inputField4.text;
        string txt5 = inputField5.text;

        temp.text = txt;
        humi.text = txt2;
        illu.text = txt3;
        tempErrorValue.text = txt4;
        humiErrorValue.text = txt5;
        Debug.Log("InputField Result!\n" + txt);
        Debug.Log("InputField2 Result!\n" + txt2);
        Debug.Log("InputField3 Result!\n" + txt3);
        Debug.Log("InputField3 Result!\n" + txt4);
        Debug.Log("InputField3 Result!\n" + txt5);

        Filed fileddata = new Filed();
        fileddata.temperature = float.Parse(txt.ToString());
        fileddata.humidity = float.Parse(txt2.ToString());
        fileddata.illuminance = float.Parse(txt3.ToString());
        fileddata.errorT = float.Parse(txt4.ToString());
        fileddata.errorH = float.Parse(txt5.ToString());

        string json = JsonUtility.ToJson(fileddata);
        StartCoroutine(Upload("http://192.168.0.2:5000/setEnv",json));
        Debug.Log(json);
        // GetMemberData();
    }

    private void Function_InputField(string _data)
    {
        string txt = _data;
        Debug.Log("InputField Typing!\n" + _data);
    }
    private void Function_InputField_EndEdit(string _data)
    {
        string txt = _data;
        Debug.LogError("InputField EndEdit!\n" + _data);
    }

    private void ResetFunction_UI()
    {
        button.onClick.RemoveAllListeners();
        inputField.placeholder.GetComponent<Text>().text = "온도를 입력하세요";
        inputField.onValueChanged.RemoveAllListeners();
        inputField.onEndEdit.RemoveAllListeners();
        inputField.contentType = InputField.ContentType.Standard;
        inputField.lineType = InputField.LineType.MultiLineNewline;
        inputField2.placeholder.GetComponent<Text>().text = "습도를 입력하세요";
        inputField2.onValueChanged.RemoveAllListeners();
        inputField2.onEndEdit.RemoveAllListeners();
        inputField2.contentType = InputField.ContentType.Standard;
        inputField2.lineType = InputField.LineType.MultiLineNewline;
        inputField3.placeholder.GetComponent<Text>().text = "조도를 입력하세요";
        inputField3.onValueChanged.RemoveAllListeners();
        inputField3.onEndEdit.RemoveAllListeners();
        inputField3.contentType = InputField.ContentType.Standard;
        inputField3.lineType = InputField.LineType.MultiLineNewline;
        inputField4.placeholder.GetComponent<Text>().text = "온도 오차값 입력하세요";
        inputField4.onValueChanged.RemoveAllListeners();
        inputField4.onEndEdit.RemoveAllListeners();
        inputField4.contentType = InputField.ContentType.Standard;
        inputField4.lineType = InputField.LineType.MultiLineNewline;
        inputField5.placeholder.GetComponent<Text>().text = "습도 오차값 입력하세요";
        inputField5.onValueChanged.RemoveAllListeners();
        inputField5.onEndEdit.RemoveAllListeners();
        inputField5.contentType = InputField.ContentType.Standard;
        inputField5.lineType = InputField.LineType.MultiLineNewline;
    }
}
