import { useEffect } from "react"

function MidiPlayer({ midiUrl, audioUrl }) {
    useEffect(() => {
        const player = document.querySelector('midi-player')
        const visualizer = document.querySelector('midi-visualizer')
        if (player && visualizer) {
            player.addVisualizer(visualizer)
        }
    }, [midiUrl])

    return (
        <>
            <midi-player
                src={midiUrl}
                sound-font
            />
            <midi-visualizer
                src={midiUrl}
                type="piano-roll"
                id="myVisualizer"
            />
        </>
    )
}
export default MidiPlayer