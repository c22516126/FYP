import { useState } from "react"
import Upload from "./Upload.jsx"
function Body() {

    const [file, setFile] = useState(null)

    return (
        <>
            <div id = "bodyContent">
                <p>Insert your own audio file. The song will be returned as a piano roll with audio, showing you how the song is composed.</p>
                <Upload/>
            </div>
        </>
    )
}


export default Body