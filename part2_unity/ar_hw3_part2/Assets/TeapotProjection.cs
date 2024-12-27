using System.Collections.Generic;
using UnityEngine;

public class TeapotProjection : MonoBehaviour
{
    public Camera cameraPrefab; 
    public GameObject teapotMarkerPrefab; 
    public GameObject imagePlanePrefab; 
    public Vector3 teapotWorldPosition = new Vector3(1.0675f, -0.7009f, 3.3672f); // USB position from PLY file
    public float planeDistance = 10f; // Distance of image planes from cameras
    public List<string> imageFileNames; // List of file names for images in the Resources folder
    public int indexToShow = 0; 

    private List<GameObject> cameras = new List<GameObject>(); // 19 cameras
    private List<GameObject> imagePlanes = new List<GameObject>(); // 19 image planes
    private List<GameObject> teapotMarkers = new List<GameObject>(); // 19 teapot markers

    private void Start()
    {   
        // Load image textures from Resources folder
        List<Texture> imageTextures = LoadTexturesFromResources();

        // Hardcoded camera data from NVM file
        List<CameraParameters> camerasData = new List<CameraParameters>
        {
            // IMG_5326.JPG
            new CameraParameters
            {
                focal_length = 2836.86572266f,
                rotation = new float[,] {
                    { 0.978065873831f, -0.208297612233f, 0.00126008554858f },
                    { -0.204550012585f, -0.959290888107f, 0.194737193834f },
                    { -0.0393543216921f, -0.190723048674f, -0.980854369235f }
                },
                translation = new Vector3(-0.178082528014f, -0.462564448779f, -0.329491983042f)
            },
            // IMG_5332.JPG
            new CameraParameters
            {
                focal_length = 2843.57299805f,
                rotation = new float[,] {
                    { 0.959766437203f, 0.0273237318176f, -0.279466167817f },
                    { 0.0797934911476f, -0.980763935275f, 0.178143473819f },
                    { -0.26922283814f, -0.193276019594f, -0.943485810617f }
                },
                translation = new Vector3(-0.0652449224612f, -0.144575134923f, -0.333041787695f)
            },
            // IMG_5334.JPG
            new CameraParameters
            {
                focal_length = 2869.15478516f,
                rotation = new float[,] {
                    { 0.990939215867f, -0.0353724809273f, -0.129570137943f },
                    { -0.0382101142567f, -0.999080318456f, -0.0194776716407f },
                    { -0.128761858599f, 0.0242522587745f, -0.991379033732f }
                },
                translation = new Vector3(-0.342941462881f, 0.339458583294f, -0.748501685582f)
            },
            // IMG_5327.JPG
            new CameraParameters
            {
                focal_length = 2815.37866211f,
                rotation = new float[,] {
                    { 0.958457023657f, 0.0985572042746f, -0.267671388077f },
                    { 0.154949736419f, -0.967776851291f, 0.198495475175f },
                    { -0.239482748949f, -0.231724809935f, -0.942842679924f }
                },
                translation = new Vector3(-0.0669882334404f, -0.336374509335f, 0.103862106541f)
            },
            // IMG_5328.JPG
            new CameraParameters
            {
                focal_length = 2837.34741211f,
                rotation = new float[,] {
                    { 0.872692253504f, 0.294930864577f, -0.389130794929f },
                    { 0.355626369042f, -0.930023289265f, 0.0926656335712f },
                    { -0.334570468966f, -0.219253939294f, -0.916509974354f }
                },
                translation = new Vector3(-0.0313613647824f, -0.333172534668f, 0.54890651049f)
            },
            // IMG_5333.JPG
            new CameraParameters
            {
                focal_length = 2859.8203125f,
                rotation = new float[,] {
                    { 0.97212227615f, 0.0153573100377f, -0.233970477367f },
                    { 0.0400462678258f, -0.99406614844f, 0.101139956204f },
                    { -0.231028968708f, -0.107689859798f, -0.966968370364f }
                },
                translation = new Vector3(0.471319501832f, 0.231848151275f, -0.792482513211f)
            },
            // IMG_5319.JPG
            new CameraParameters
            {
                focal_length = 2888.30297852f,
                rotation = new float[,] {
                    { 0.986974091537f, 0.00351071928711f, -0.16084120001f },
                    { 0.0110113637814f, -0.9988915852f, 0.0457670235068f },
                    { -0.160502289553f, -0.0469417454054f, -0.985918984298f }
                },
                translation = new Vector3(0.842129757251f, 0.0331131743926f, -1.41704473468f)
            },
            // IMG_5320.JPG
            new CameraParameters
            {
                focal_length = 2844.9128418f,
                rotation = new float[,] {
                    { 0.984238875334f, -0.176847750967f, 0f },
                    { -0.176752207631f, -0.983707136192f, 0.0328783897157f },
                    { -0.00581445146401f, -0.0323600196881f, -0.999459655188f }
                },
                translation = new Vector3(-0.0454927094475f, 0.0829552700328f, -1.48704197003f)
            },
            // IMG_5318.JPG
            new CameraParameters
            {
                focal_length = 2923.09399414f,
                rotation = new float[,] {
                    { 0.99684915396f, -0.060341008297f, -0.0514998704169f },
                    { -0.0566899130846f, -0.995961128893f, 0.0696325058777f },
                    { -0.0554934988828f, -0.0664934478582f, -0.996242850781f }
                },
                translation = new Vector3(0.714616169438f, 0.108129614401f, -2.26809359135f)
            },
            // IMG_5331.JPG
            new CameraParameters
            {
                focal_length = 2881.38378906f,
                rotation = new float[,] {
                    { 0.957312105602f, -0.284849853859f, 0.0491335570031f },
                    { -0.287224099938f, -0.918292255299f, 0.272475137138f },
                    { -0.0324955972472f, -0.274956153217f, -0.960907671376f }
                },
                translation = new Vector3(-0.733173668925f, -0.110338542168f, -0.754766092824f)
            },
            // IMG_5322.JPG
            new CameraParameters
            {
                focal_length = 2846.19384766f,
                rotation = new float[,] {
                    { 0.95108534678f, -0.280404893956f, 0.129651510143f },
                    { -0.300529702611f, -0.936996533218f, 0.178099932261f },
                    { 0.0715430287822f, -0.208352449537f, -0.975433892188f }
                },
                translation = new Vector3(-0.89150881299f, -0.11593452198f, -1.23445028632f)
            },
            // IMG_5321.JPG
            new CameraParameters
            {
                focal_length = 2879.71850586f,
                rotation = new float[,] {
                    { 0.978118674481f, -0.198142381328f, 0.0634375193955f },
                    { -0.20804007459f, -0.93444172752f, 0.289030468275f },
                    { 0.00200940712477f, -0.295903516535f, -0.955216274576f }
                },
                translation = new Vector3(-0.199567869306f, -0.556868814597f, -1.27281416111f)
            },
            // IMG_5324.JPG
            new CameraParameters
            {
                focal_length = 2926.328125f,
                rotation = new float[,] {
                    { 0.953966420787f, -0.288984565759f, 0.0802264098765f },
                    { -0.297012539145f, -0.947428029676f, 0.119012267317f },
                    { 0.041616015714f, -0.13736200608f, -0.989646553684f }
                },
                translation = new Vector3(-0.877888461974f, -0.0236202639397f, -2.59210787186f)
            },
            // IMG_5323.JPG
            new CameraParameters
            {
                focal_length = 2875.38793945f,
                rotation = new float[,] {
                    { 0.916016404619f, -0.35287518566f, 0.190768267461f },
                    { -0.356863356057f, -0.934048798887f, -0.0142048761067f },
                    { 0.183199544664f, -0.0550661442684f, -0.981532108241f }
                },
                translation = new Vector3(-1.19000728687f, 0.526446484953f, -1.30392542547f)
            },
            // IMG_5329.JPG
            new CameraParameters
            {
                focal_length = 2857.26171875f,
                rotation = new float[,] {
                    { 0.931398935676f, -0.330186654238f, 0.153207902667f },
                    { -0.34934638349f, -0.929088602583f, 0.121457015159f },
                    { 0.102239975039f, -0.166647478685f, -0.980701599381f }
                },
                translation = new Vector3(-1.49294786541f, 0.676084231296f, -0.65155518049f)
            },
            // IMG_5330.JPG
            new CameraParameters
            {
                focal_length = 2811.3112793f,
                rotation = new float[,] {
                    { 0.88722291726f, -0.385863669223f, 0.252873287492f },
                    { -0.391615985094f, -0.919661731676f, -0.029316815137f },
                    { 0.243869915208f, -0.0730185612312f, -0.967054776881f }
                },
                translation = new Vector3(-1.2973016803f, 1.0344078034f, -0.532247254985f)
            },
            // IMG_5316.JPG
            new CameraParameters
            {
                focal_length = 2928.49121094f,
                rotation = new float[,] {
                    { 0.909385629599f, -0.339063179787f, 0.240944347397f },
                    { -0.344072660464f, -0.938678114196f, -0.0223134591465f },
                    { 0.233734671133f, -0.0626106805132f, -0.970282301763f }
                },
                translation = new Vector3(-0.703526756486f, 0.464820783258f, -3.15883150318f)
            },
            // IMG_5325.JPG
            new CameraParameters
            {
                focal_length = 2998.23217773f,
                rotation = new float[,] {
                    { 0.954681663264f, -0.274429909433f, 0.115200066545f },
                    { -0.276003607862f, -0.961153451009f, -0.00237508421905f },
                    { 0.111376760976f, -0.0295282180946f, -0.99333952523f }
                },
                translation = new Vector3(-0.690341702511f, 0.302343210058f, -4.13599570357f)
            },
            // IMG_5315.JPG
            new CameraParameters
            {
                focal_length = 2927.61621094f,
                rotation = new float[,] {
                    { 0.863998918638f, -0.389931812649f, 0.318526518859f },
                    { -0.399118723329f, -0.916076457439f, -0.0388328638605f },
                    { 0.30693693324f, -0.0935783851867f, -0.947118286498f }
                },
                translation = new Vector3(-1.21416451002f, 0.436194237431f, -3.05026363423f)
            }
        };
        
        // Show the camera parameters and camera number for the selected index
        Debug.Log($"Camera {indexToShow + 1} Parameters:");
        Debug.Log($"Focal Length: {camerasData[indexToShow].focal_length}");
        Debug.Log("Rotation Matrix:");
        for (int i = 0; i < 3; i++)
        {
            string row = "";
            for (int j = 0; j < 3; j++)
            {
            row += camerasData[indexToShow].rotation[i, j].ToString("F6") + " ";
            }
            Debug.Log(row);
        }
        Debug.Log($"Translation: {camerasData[indexToShow].translation}");

        for (int i = 0; i < camerasData.Count; i++)
        {
            Camera unityCamera = Instantiate(cameraPrefab);
            GameObject imagePlane = Instantiate(imagePlanePrefab);
            GameObject teapotMarker = Instantiate(teapotMarkerPrefab);

            // Set up the camera, image plane, and teapot marker
            SetupCamera(camerasData[i], i, imageTextures, unityCamera, imagePlane, teapotMarker);

            // Store the created objects for enabling/disabling
            cameras.Add(unityCamera.gameObject);
            imagePlanes.Add(imagePlane);
            teapotMarkers.Add(teapotMarker);
        }

        // Enable only the selected index, deactivate others
        for (int i = 0; i < cameras.Count; i++)
        {
            bool isActive = (i == indexToShow);
            cameras[i].SetActive(isActive);
            imagePlanes[i].SetActive(isActive);
            teapotMarkers[i].SetActive(isActive);
        }

    }

    private List<Texture> LoadTexturesFromResources()
    {
        List<Texture> textures = new List<Texture>();
        foreach (string fileName in imageFileNames)
        {
            Texture texture = Resources.Load<Texture>(fileName);
            if (texture != null)
            {
                textures.Add(texture);
                Debug.Log($"Successfully loaded texture: {fileName}");
            }
            else
            {
                Debug.LogError($"Failed to load texture: {fileName}");
            }
        }
        return textures;
    }

    // Set up the camera, image plane, and teapot marker
    private void SetupCamera(CameraParameters cam, int index, List<Texture> imageTextures, Camera unityCamera, GameObject imagePlane, GameObject teapotMarker)
    {
        // Set camera position and rotation
        unityCamera.transform.position = cam.translation;
        Matrix4x4 rotationMatrix = MatrixFrom2DArray(cam.rotation);
        unityCamera.transform.rotation = RotationFromMatrix(rotationMatrix);

        // Invert the camera's Z direction to align with Unity's coordinate system
        unityCamera.transform.Rotate(0f, 0f, 180f);

        float nearClip = unityCamera.nearClipPlane = 0.1f; // Near clip plane means the camera can't see anything closer than 0.1 units
        float farClip = unityCamera.farClipPlane = 100f;   // Far clip plane means the camera can't see anything further than 100 units
        float imageWidth = 3264;  
        float imageHeight = 2448; 
        unityCamera.aspect = imageWidth / imageHeight; // Aspect ratio of the image

        // unityCamera.fieldOfView = 60f; 

        // Calculate FOV from focal length
        float sensorHeight = 2448; 
        float fovY = 2f * Mathf.Atan((sensorHeight/2) / cam.focal_length) * Mathf.Rad2Deg;
        unityCamera.fieldOfView = fovY;

        // Position the image plane
        imagePlane.transform.position = unityCamera.transform.position + unityCamera.transform.forward * planeDistance;
        imagePlane.transform.LookAt(unityCamera.transform.position);
        imagePlane.transform.Rotate(0f, 180f, 0f);

        // Scale the image plane to match the camera's FOV
        float planeHeight = 2f * planeDistance * Mathf.Tan(unityCamera.fieldOfView * 0.5f * Mathf.Deg2Rad);
        float planeWidth = planeHeight * unityCamera.aspect;
        imagePlane.transform.localScale = new Vector3(planeWidth, planeHeight, 1f);

        // Assign texture to the image plane
        if (index < imageTextures.Count)
        {
            Renderer renderer = imagePlane.GetComponent<Renderer>();
            renderer.material = new Material(Shader.Find("Unlit/Texture"));
            renderer.material.mainTexture = imageTextures[index];
        }

        // Project and position the teapot marker
        Vector3 projectedTeapot = unityCamera.WorldToScreenPoint(teapotWorldPosition);
        teapotMarker.transform.position = unityCamera.ScreenToWorldPoint(
            new Vector3(projectedTeapot.x, projectedTeapot.y, planeDistance)
        );

        // // Scale the teapot to have a bounding box of 15 cm length
        // Bounds teapotBounds = teapotMarker.GetComponent<Renderer>().bounds;
        // float maxDimension = Mathf.Max(teapotBounds.size.x, teapotBounds.size.y, teapotBounds.size.z);
        // float scaleFactor = 0.15f / maxDimension; // 0.15 meters (15 cm)
        // teapotMarker.transform.localScale *= scaleFactor;

        teapotMarker.transform.LookAt(imagePlane.transform.position); 
        teapotMarker.transform.Rotate(0f, 0f, 90f); // Rotate the teapot marker to face the image plane

        // Debugging
        Debug.Log($"Camera {index + 1}: Teapot projected at {teapotMarker.transform.position}");
        Debug.Log($"Camera {index + 1}: Image plane at {imagePlane.transform.position}");
        Debug.Log($"Camera {index + 1}: Camera at {unityCamera.transform.position}");
    }


    public void ShowSetup(int index)
    {
        // Clamp the index to valid values
        if (index < 0 || index >= cameras.Count)
        {
            Debug.LogError($"Invalid index: {index}. Must be between 0 and {cameras.Count - 1}");
            return;
        }

        // Enable only the selected setup
        for (int i = 0; i < cameras.Count; i++)
        {
            bool isActive = (i == index);
            cameras[i].SetActive(isActive);
            imagePlanes[i].SetActive(isActive);
            teapotMarkers[i].SetActive(isActive);
        }

        indexToShow = index;
    }

    private Matrix4x4 MatrixFrom2DArray(float[,] array)
    {
        Matrix4x4 matrix = Matrix4x4.identity;
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                matrix[i, j] = array[i, j];
            }
        }
        return matrix;
    }

    private Quaternion RotationFromMatrix(Matrix4x4 matrix)
    {
        return Quaternion.LookRotation(
            matrix.GetColumn(2), // Forward vector
            matrix.GetColumn(1)  // Up vector
        );
    }

}

[System.Serializable]
public class CameraParameters
{
    public float focal_length;
    public float[,] rotation;
    public Vector3 translation;
}
