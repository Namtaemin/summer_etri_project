using System.Collections;
using System.Collections.Generic;
using System.Net;
using System.IO;
using UnityEngine;
using UnityEngine.Networking;
using UnityEngine.UI;

[System.Serializable]
public class FutureLoginDataValue{
}

[System.Serializable]
public class FutureDataValue{
    public float temperature;
    public float humidity;
    public float illuminance;
}

// [System.Serializable]
// public class FutureLoginDataValue1{
// }

// [System.Serializable]
// public class FutureDataValue1{
//     public float temperature;
//     public float humidity;
//     public float illuminance;
// }

// [System.Serializable]
// public class FutureLoginDataFuzzy{
// }

// [System.Serializable]
// public class FutureDataFuzzy{
//     public float temperatureFuzzy;
//     public float humidityFuzzy;
//     public float illuminanceFuzzy;
// }

[System.Serializable]
public class Times
{
    public string iter;
    public float temperature;
    public float humidity;
    public float illuminance;
}

public class DropdownExample : MonoBehaviour
{

    [Header("Dropdown")]
    public Dropdown dropdown;
    public Text TemperatureText;
    public Text HumidityText;
    public Text IlluminanceText;
    // public Text FuzzyTemperatureText;
    // public Text FuzzyHumidityText;
    // public Text FuzzyIlluminanceText;

    public string BaseURL = "http://192.168.0.2:5000/getPrediction";
    // public string BaseURL2 = "http://192.168.0.2:5000/getFuzzy";
    void Start()
    {
        GetData();
        SetDropdownOptionsExample();
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

            Times time = new Times();
            
            time = JsonUtility.FromJson<Times>(request.downloadHandler.text);
            string fileName = "Myjson";
            string path = Application.dataPath + "/Resource/Json/"+ fileName + ".Json";

            File.WriteAllText(path, request.downloadHandler.text);
            
            string filePath = "Assets/Myjson.json";
            string jsonon =File.ReadAllText(filePath);
            //Times time = JsonUtility.FromJson<Times>(jsonon);
            
            Debug.Log(time.temperature);
            Debug.Log(time.humidity);
            Debug.Log(time.illuminance);            
            TemperatureText.text = time.temperature.ToString();
            HumidityText.text = time.humidity.ToString();
            IlluminanceText.text = time.illuminance.ToString();

        }
    }


    // private FutureDataFuzzy FuzzyGetData()
    // {
    //     FutureLoginDataFuzzy data = new FutureLoginDataFuzzy();

    //     string str = JsonUtility.ToJson(data);
    //     var bytes = System.Text.Encoding.UTF8.GetBytes(str);

    //     HttpWebRequest request = (HttpWebRequest)WebRequest.Create(BaseURL2);
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
    //     string json = reader.ReadToEnd();
    //     FutureDataFuzzy info = JsonUtility.FromJson<FutureDataFuzzy>(json);

    //     Debug.Log("temperature : "+info.temperatureFuzzy);
    //     Debug.Log("humidity : "+info.humidityFuzzy);
    //     Debug.Log("illuminance : "+info.illuminanceFuzzy);

    //     TemperatureText.text= info.temperatureFuzzy.ToString();
    //     HumidityText.text= info.humidityFuzzy.ToString();
    //     IlluminanceText.text= info.illuminanceFuzzy.ToString();
        
    //     return info;
    // }


    private FutureDataValue GetData()
    {
        FutureLoginDataValue data = new FutureLoginDataValue();

        string str = JsonUtility.ToJson(data);
        var bytes = System.Text.Encoding.UTF8.GetBytes(str);

        HttpWebRequest request = (HttpWebRequest)WebRequest.Create(BaseURL);
        request.Method = "POST";
        request.ContentType = "application/json";
        request.ContentLength = bytes.Length;

        using(var stream = request.GetRequestStream())
        {
            stream.Write(bytes, 0, bytes.Length);
            stream.Flush();
            stream.Close();
        }

        HttpWebResponse response = (HttpWebResponse)request.GetResponse();
        StreamReader reader = new StreamReader(response.GetResponseStream());
        string json = reader.ReadToEnd();
        FutureDataValue info = JsonUtility.FromJson<FutureDataValue>(json);

        Debug.Log("temperature : "+info.temperature);
        Debug.Log("humidity : "+info.humidity);
        Debug.Log("illuminance : "+info.illuminance);

        TemperatureText.text= info.temperature.ToString();
        HumidityText.text= info.humidity.ToString();
        IlluminanceText.text= info.illuminance.ToString();
        

        return info;
    }

    private void SetDropdownOptionsExample() // Dropdown 목록 생성    
    {   dropdown.options.Clear();
        
        for(int i = 0; i < 13; i++) //1부터 10까지        
        {
            Dropdown.OptionData option = new Dropdown.OptionData();
            option.text = i.ToString() +"시간 후";
            dropdown.options.Add(option);
        }
    }
        // Update is called once per frame
    void Update()
    {  
    }

    public void SelectButton(int iterator)
    {
        
        Times time = new Times();
        time.iter = iterator.ToString();
        string json = JsonUtility.ToJson(time);
        iterator = GetComponent<Dropdown>().value;

        switch(iterator)
        {
            case 0:
                Debug.Log("iterator"+ iterator);
                StartCoroutine(Upload("http://192.168.0.2:5000/getPrediction",json));
                break;
            case 1:
                Debug.Log("iterator"+ iterator);
                StartCoroutine(Upload("http://192.168.0.2:5000/getPrediction",json));
                break;
            
            case 2:
                Debug.Log("iterator"+ iterator);
                StartCoroutine(Upload("http://192.168.0.2:5000/getPrediction",json));

                break;

            case 3:
                Debug.Log("iterator"+ iterator);
                StartCoroutine(Upload("http://192.168.0.2:5000/getPrediction",json));
                break;

            case 4:
                Debug.Log("iterator"+ iterator);
                StartCoroutine(Upload("http://192.168.0.2:5000/getPrediction",json));               
                break;

            case 5:
                Debug.Log("iterator"+ iterator);
                StartCoroutine(Upload("http://192.168.0.2:5000/getPrediction",json));              
                break;

            case 6:
                Debug.Log("iterator"+ iterator);
                StartCoroutine(Upload("http://192.168.0.2:5000/getPrediction",json));             
                break;

            case 7:
                Debug.Log("iterator"+ iterator);
                StartCoroutine(Upload("http://192.168.0.2:5000/getPrediction",json));              
                break;

            case 8:
                Debug.Log("iterator"+ iterator);
                StartCoroutine(Upload("http://192.168.0.2:5000/getPrediction",json));             
                break;

            case 9:
                Debug.Log("iterator"+ iterator);
                StartCoroutine(Upload("http://192.168.0.2:5000/getPrediction",json));              
                break;

            case 10:
                Debug.Log("iterator"+ iterator);
                StartCoroutine(Upload("http://192.168.0.2:5000/getPrediction",json));              
                break;

            case 11:
                Debug.Log("iterator"+ iterator);
                StartCoroutine(Upload("http://192.168.0.2:5000/getPrediction",json));               
                break;

        }
        
        
    }
}
