HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
    <meta http-equiv="content-type" content="text/html; charset=UTF-8">
    <title>Canopy API Spec</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <link rel="shortcut icon" href="https://polybit-apps.s3.amazonaws.com/stdlib/users/pinecone/profile/image.png">
    <style>
        body {
            margin: 0;
            padding: 0;
        }
    </style>
    <style data-styled="" data-styled-version="4.4.1"></style>
</head>
<body>
    <div id="redoc-container"></div>
    <title>Redoc</title>
    <script src="https://cdn.jsdelivr.net/npm/redoc/bundles/redoc.standalone.js"> </script>
    <script>
        var spec = %s;
        Redoc.init(spec, {}, document.getElementById("redoc-container"));
    </script>
</body>
</html>
"""  # noqa: E501
