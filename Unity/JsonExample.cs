using System.Collections;
using System.Collections.Generic;
using System.Net;
using System.IO;
using UnityEngine;
using UnityEngine.Networking;
using UnityEngine.UI;


[System.Serializable]
public class LoginData{
}
[System.Serializable]
public class SensorData{
    public float temperature;
    public float humidity;
    public float illuminance;
    public string dataTime;
}


public class JsonExample : MonoBehaviour
{
    public Text TemperatureText;
    public Text HumidityText;
    public Text IlluminanceText;
    // public Text TemperatureFuzzy;
    // public Text HumidityFuzzy;
    // public Text IlluminanceFuzzy;

    public string BaseURL = "http://192.168.0.2:5000/getValue";
    // Start is called before the first frame update
    void Start()
    {
        GetMemberData();
        

    }
    private SensorData GetMemberData()
    {
        LoginData data = new LoginData();

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
        SensorData info = JsonUtility.FromJson<SensorData>(json);
        
        Debug.Log("temperature : "+info.temperature);
        Debug.Log("humidity : "+info.humidity);
        Debug.Log("illuminance : "+info.illuminance);

        TemperatureText.text= info.temperature.ToString();
        HumidityText.text= info.humidity.ToString();
        IlluminanceText.text= info.illuminance.ToString();

        SensorData mydata = new SensorData();
        mydata.temperature = float.Parse(info.temperature.ToString());
        mydata.humidity = float.Parse(info.humidity.ToString());
        mydata.illuminance = float.Parse(info.illuminance.ToString());
        // SensorData user1 = new SensorData
        // {
        //     temperature = float.Parse(info.temperature.ToString()),
        //     humidity = float.Parse(info.humidity.ToString()),
        //     illuminance = float.Parse(info.illuminance.ToString())
        // };
        // Users user_arr = new Users();
        // user_arr.users.Add(user1);
        
        string json2 = JsonUtility.ToJson(mydata);
        Debug.Log(json2);
        StartCoroutine(Upload("http://192.168.0.2:5000/raspberry",json2));

        return info;
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
            if (request.isNetworkError || request.isHttpError || request.isNetworkError)
            {
                print("Error: " + request.error);
            }
            else
            {
                Debug.Log(request.downloadHandler.text);
            }
            
        }
    }
        // Update is called once per frame
    void Update()
    {  
        TemperatureText = GameObject.Find("현재온도텍스트").GetComponent<Text>();
        HumidityText = GameObject.Find("현재습도텍스트").GetComponent<Text>();
        IlluminanceText = GameObject.Find("현재조도텍스트").GetComponent<Text>();

    }

    
}
