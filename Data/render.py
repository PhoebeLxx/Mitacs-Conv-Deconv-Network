#!/usr/bin/env python

"""
"""

import vtk
import numpy as np

def main():
    args = get_program_parameters()


    # Set some named colours, so we can use them later
    colors = vtk.vtkNamedColors()
    colors.SetColor("BkgColor", [0, 0, 0, 255])
    colors.SetColor("SurfaceColor", [200, 200, 200, 255])

    # Process all the command line arguments
    fileName = args.filename
    dim = [int(_) for _ in args.dimensions.split(',')]
    output = args.output
    isovalue = float(args.isovalue)

    # read the raw data
    with open(fileName, 'rb') as dataHandle:
        data = dataHandle.read()

    # For VTK to be able to use the data, it must be stored as a VTK-image. This can be done by the vtkImageImport-class which
    # imports raw data and stores it.
    dataImporter = vtk.vtkImageImport()

    # Tell VTK to copy the raw data 
    dataImporter.CopyImportVoidPointer(data, len(data))
    
    # The type of the newly imported data is set to unsigned char (uint8)
    dataImporter.SetDataScalarTypeToUnsignedChar()
    
    # Because the data that is imported only contains an intensity value (it isnt RGB-coded or someting similar), the importer
    # must be told this is the case.
    dataImporter.SetNumberOfScalarComponents(1)

    # The following two functions describe how the data is stored and the dimensions of the array it is stored in. For this
    # simple case, all axes are of length 75 and begins with the first element. For other data, this is probably not the case.
    # I have to admit however, that I honestly dont know the difference between SetDataExtent() and SetWholeExtent() although
    # VTK complains if not both are used.
    dataImporter.SetDataExtent(0, dim[0]-1, 0, dim[1]-1, 0, dim[2]-1)
    dataImporter.SetWholeExtent(0, dim[0]-1, 0, dim[1]-1, 0, dim[2]-1)



    # Create the renderer, the render window, and the interactor. The renderer
    # draws into the render window, the interactor enables mouse- and
    # keyboard-based interaction with the scene.
    ren = vtk.vtkRenderer()
    renWin = vtk.vtkRenderWindow()
    renWin.AddRenderer(ren)
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)


    # Setup an isosurface extraction
    surfaceExtractor = vtk.vtkMarchingCubes()
    surfaceExtractor.SetInputConnection(dataImporter.GetOutputPort())
    surfaceExtractor.SetValue(0, isovalue)

    surfaceStripper = vtk.vtkStripper()
    surfaceStripper.SetInputConnection(surfaceExtractor.GetOutputPort())

    surfaceMapper = vtk.vtkPolyDataMapper()
    surfaceMapper.SetInputConnection(surfaceStripper.GetOutputPort())
    surfaceMapper.ScalarVisibilityOff()

    # Set the surface and shading properties of the extracted isosurface
    surface = vtk.vtkActor()
    surface.SetMapper(surfaceMapper)
    surface.GetProperty().SetDiffuseColor(colors.GetColor3d("SurfaceColor"))
    surface.GetProperty().SetSpecular(.3)
    surface.GetProperty().SetSpecularPower(20)
    surface.GetProperty().SetOpacity(1)
 
    # Assign the actor to the renderer
    ren.AddActor(surface)

    # Set up an initial view of the volume.  The focal point will be the
    # center of the volume, and the camera position will be 400mm to the
    # patient's left (which is our right).
    camera = ren.GetActiveCamera()
    c = surface.GetCenter()
    camera.SetViewUp(0, 0, -1)
    camera.SetPosition(c[0], c[1] - 300, c[2])
    camera.SetFocalPoint(c[0], c[1], c[2])
    camera.Azimuth(0.0)
    camera.Elevation(0.0)

    # Set a background color for the renderer
    ren.SetBackground(colors.GetColor3d("BkgColor"))

    # Increase the size of the render window
    renWin.SetSize(512, 512)
    renWin.Render()

    writer = vtk.vtkPNGWriter()
    windowto_image_filter = vtk.vtkWindowToImageFilter()
    windowto_image_filter.SetInput(renWin)
    windowto_image_filter.SetScale(1)  # image quality

    windowto_image_filter.SetInputBufferTypeToRGB()
    # Read from the front buffer.
    windowto_image_filter.ReadFrontBufferOff()
    windowto_image_filter.Update()

    writer.SetFileName(output)
    writer.SetInputConnection(windowto_image_filter.GetOutputPort())
    writer.Write()
    
    # Uncomment this to interact with the data.
    # iren.Start()


def get_program_parameters():
    import argparse
    description = 'Read a volume dataset and renders its contour.'
    epilogue = '''
    Derived from VTK/Examples/Cxx/Medical2.cxx
    '''
    parser = argparse.ArgumentParser(description=description, epilog=epilogue,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('filename', help='input.raw')
    parser.add_argument('dimensions', help='128,128,128')
    parser.add_argument('output', help="output.png")
    parser.add_argument('isovalue', help="5.0")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()

