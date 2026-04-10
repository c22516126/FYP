import mirdata

maestro = mirdata.initialize('maestro')

maestro.download(["index"])
print(maestro.index_path)