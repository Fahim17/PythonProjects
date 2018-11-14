from wand.image import Image as wi

try:
	pdf = wi(filename = "/home/fahim/Py_project/SideProject/sample.pdf", resolution = 300)
	pdfImage = pdf.convert("jpeg");
	i=1

	for img in pdfImage.sequence:
		page = wi(image=img)
		page.save(filename = str(i)+".jpg")
		i+=1
	
except TypeError as e:
	print(e)