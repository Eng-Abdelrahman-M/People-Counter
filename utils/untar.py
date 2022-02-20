import tarfile

class Untar:
    @staticmethod
    def untar_bz2(source,destinattion):
        tar = tarfile.open(source, "r:bz2")  
        tar.extractall(destinattion)
        tar.close()
