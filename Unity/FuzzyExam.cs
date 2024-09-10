using System.Collections;
using System.Collections.Generic;
using System.Net;
using System.IO;
using UnityEngine;
using UnityEngine.Networking;
using UnityEngine.UI;

[System.Serializable]
public class FuzzyData{
    public float controlT;
    public float controlH;
    public float controlI;
    public string stateT;
    public string stateH;
    public string stateI;
    
}

public class FuzzyExam : MonoBehaviour
{
    public Text TemperatureText;
    public Text HumidityText;
    public Text IlluminanceText;
    public Button button;

    public string BaseURL = "http://192.168.0.2:5000/getControl";
    // Start is called before the first frame update
    void Start()
    {
        button.onClick.AddListener(Function_Button);
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

            FuzzyData time = new FuzzyData();
            
            time = JsonUtility.FromJson<FuzzyData>(request.downloadHandler.text);
            string fileName = "Myfuzzyjson";
            string path = Application.dataPath + "/resource/Json/"+ fileName + ".Json";

            File.WriteAllText(path, request.downloadHandler.text);
            
            string filePath = "Assets/resource/Json/Myfuzzyjson.json";
            string jsonon =File.ReadAllText(filePath);
            //Times time = JsonUtility.FromJson<Times>(jsonon);
            
            Debug.Log(time.stateT);
            Debug.Log(time.stateH);
            Debug.Log(time.stateI);            
            TemperatureText.text = time.stateT.ToString();
            HumidityText.text = time.stateH.ToString();
            IlluminanceText.text = time.stateI.ToString();

        }
    }

    private void Function_Button()
    {
        FuzzyData time = new FuzzyData();
        string json = JsonUtility.ToJson(time);
        StartCoroutine(Upload("http://192.168.0.2:5000/getControl",json));

    }
        // Update is called once per frame
    void Update()
    {
 
        // TemperatureText = GameObject.Find("온도퍼지").GetComponent<Text>();
        // HumidityText = GameObject.Find("습도퍼지").GetComponent<Text>();
        // IlluminanceText = GameObject.Find("조도퍼지").GetComponent<Text>();
    }

    
}
