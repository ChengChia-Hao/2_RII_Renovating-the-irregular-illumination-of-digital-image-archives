import cv2
import copy
import numpy as np
# from numba import njit

# @njit
def EdgeImg(aImg):
    iHeight, iWidth = aImg.shape

    aKernel = np.array([
        [
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ],
        [
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ],
        [
            [-1, -1, -1],
            [-1, 8, -1],
            [-1, -1, -1]
        ],
        [
            [1, 1, 1],
            [1, -8, 1],
            [1, 1, 1]
        ],
        [
            [-1, 2, -1],
            [2, -4, 2],
            [-1, 2, -1]
        ]
    ])

    aEdgeImg = np.zeros((len(aKernel), iHeight, iWidth))

    for iKernelIndex in range(len(aKernel)):
        for iHeightIndex in range(iHeight - 2):
            for iWidthIndex in range(iWidth - 2):
                aEdgeImg[iKernelIndex, iHeightIndex, iWidthIndex] = abs(np.sum(aImg[iHeightIndex : iHeightIndex + 3, iWidthIndex : iWidthIndex + 3] * aKernel[iKernelIndex]))

                if 0 > aEdgeImg[iKernelIndex, iHeightIndex, iWidthIndex]:
                    aEdgeImg[iKernelIndex, iHeightIndex, iWidthIndex] = 0
                if 255 < aEdgeImg[iKernelIndex, iHeightIndex, iWidthIndex]:
                    aEdgeImg[iKernelIndex, iHeightIndex, iWidthIndex] = 255

    return aEdgeImg

# @njit
def EdgeAvgImg(aEdgeImg):
    iKernel, iHeight, iWidth = aEdgeImg.shape

    aEdgeAvgImg = np.zeros((iHeight, iWidth))

    for iHeightIndex in range(iHeight):
        for iWidthIndex in range(iWidth):
            for iKernelIndex in range(iKernel):
                aEdgeAvgImg[iHeightIndex, iWidthIndex] += aEdgeImg[iKernelIndex, iHeightIndex, iWidthIndex]

            aEdgeAvgImg[iHeightIndex, iWidthIndex] = int(aEdgeAvgImg[iHeightIndex, iWidthIndex] / iKernel)
    
    return aEdgeAvgImg

# @njit
def BinaryImg(aEdgeAvgImg, iThreshold):
    iHeight, iWidth = aEdgeAvgImg.shape

    aBinaryImg = np.zeros((iHeight, iWidth))

    for iHeightIndex in range(iHeight):
        for iWidthIndex in range(iWidth):
            if iThreshold <= aEdgeAvgImg[iHeightIndex, iWidthIndex]:
                aBinaryImg[iHeightIndex, iWidthIndex] = 255
            else:
                aBinaryImg[iHeightIndex, iWidthIndex] = 0

    return aBinaryImg

# @njit
def CheckNeighbors(aBinaryImg, iNeighborHeight, iNeighborWidth):
    if 255 == aBinaryImg[iNeighborHeight + 1, iNeighborWidth]:
        return iNeighborHeight + 1, iNeighborWidth
    elif 255 == aBinaryImg[iNeighborHeight - 1, iNeighborWidth]:
        return iNeighborHeight - 1, iNeighborWidth
    elif 255 == aBinaryImg[iNeighborHeight, iNeighborWidth + 1]:
        return iNeighborHeight, iNeighborWidth + 1
    elif 255 == aBinaryImg[iNeighborHeight + 1, iNeighborWidth - 1]:
        return iNeighborHeight, iNeighborWidth - 1

    return 0, 0

# @njit
def MarkedImg(aMarkedImg, aBinaryImg):

    iHeight, iWidth = aBinaryImg.shape

    for iHeightIndex in range(iHeight):
        for iWidthIndex in range(iWidth):
            bFlag = True

            if 255 == aBinaryImg[iHeightIndex, iWidthIndex]:
                aNeighbors = []
                iUp = iDown = iHeightIndex
                iLeft = iRight = iWidthIndex
                aBinaryImg[iHeightIndex, iWidthIndex] = 0
                
                aNeighbors.append((iHeightIndex, iWidthIndex))

                while bFlag:
                    bFlag = False

                    for iNeighborHeightIndex, iNeighborWidthIndex in aNeighbors:
                        iNewNeighborHeight, iNewNeighborWidth = CheckNeighbors(aBinaryImg, iNeighborHeightIndex, iNeighborWidthIndex)
                        if 0 != iNewNeighborHeight and 0 != iNewNeighborWidth:
                            bFlag = True
                            aBinaryImg[iNewNeighborHeight, iNewNeighborWidth] = 0
                            
                            if iUp > iNewNeighborHeight:
                                iUp = iNewNeighborHeight

                            if iDown < iNewNeighborHeight:
                                iDown = iNewNeighborHeight

                            if iLeft > iNewNeighborWidth:
                                iLeft = iNewNeighborWidth

                            if iRight < iNewNeighborWidth:
                                iRight = iNewNeighborWidth

                            aNeighbors.append((iNewNeighborHeight, iNewNeighborWidth))

                for iNeighborHeightIndex in range(iUp, iDown + 1):
                    for iNeighborWidthIndex in range(iLeft, iRight + 1):
                        aMarkedImg[iNeighborHeightIndex, iNeighborWidthIndex] = 0

    return aMarkedImg

# @njit
def LDIImg(aLDIImg, aMarkedImg):
    iHeight, iWidth = aMarkedImg.shape

    iMarkedSetions = 0
    Pvf = 0
    Pvl = 0
    iMarkedMinHeight = 0
    iMarkedMaxHeight = 0

    for iWidthIndex in range(iWidth):
        for iHeightIndex in range(iHeight):
            if (0 == aMarkedImg[iHeightIndex, iWidthIndex]):
                iMarkedSetions += 1
                iMarkedMaxHeight = iHeightIndex

                if (1 == iMarkedSetions):
                    iMarkedMinHeight = iHeightIndex

                    if (iHeightIndex - 1 < 0):
                        Pvf = int(aLDIImg[iHeightIndex, iWidthIndex])
                    else:
                        Pvf = int(aLDIImg[iHeightIndex - 1, iWidthIndex])
            else:
                if (0 < iMarkedSetions):
                    kPixel = 0

                    if (iHeightIndex + 1 > iHeight - 1):
                        Pvl = int(aLDIImg[iHeightIndex, iWidthIndex])
                    else:
                        Pvl = int(aLDIImg[iHeightIndex + 1, iWidthIndex])

                    for iLDIIndex in range(iMarkedMinHeight, iMarkedMaxHeight + 1):
                        kPixel += 1

                        iLDI = int(Pvf + (Pvl - Pvf) / iMarkedSetions * kPixel)
                        if (255 < iLDI):
                            iLDI = 255
                        if (0 >= iLDI):
                            iLDI = 0

                        aLDIImg[iLDIIndex, iWidthIndex] = iLDI

                iMarkedSetions = 0

    return aLDIImg

# @njit
def BalacedImg(aBalacedImg, aLDIImg, aMarkedImg, iBalance):
    iHeight, iWidth = aBalacedImg.shape

    for iHeightIndex in range(iHeight):
        for iWidthIndex in range(iWidth):
            iBalacingRadiance = 255

            if 0 == aMarkedImg[iHeightIndex, iWidthIndex]:
                if 0 == aLDIImg[iHeightIndex, iWidthIndex]:
                    aLDIImg[iHeightIndex, iWidthIndex] = 1

                iBalacingRadiance = int((iBalance / int(aLDIImg[iHeightIndex, iWidthIndex])) * int(aBalacedImg[iHeightIndex, iWidthIndex]))
                if 255 < iBalacingRadiance:
                    iBalacingRadiance = 255

            aBalacedImg[iHeightIndex, iWidthIndex] = iBalacingRadiance

    return aBalacedImg

# @njit
def ExposureDegree(aImg):
    iHeight, iWidth = aImg.shape

    bExposure = False
    iRange = 4
    aAverage = []

    for iHeightIndex in range(0, iHeight - iRange, iRange):
        for iWidthIndex in range(0, iWidth - iRange, iRange):
            iAveragePixel = (int(aImg[iHeightIndex, iWidthIndex]) + int(aImg[iHeightIndex, iWidthIndex + iRange - 1]) + int(aImg[iHeightIndex + iRange - 1, iWidthIndex]) + int(aImg[iHeightIndex + iRange - 1, iWidthIndex + iRange - 1])) / iRange
            if 200 <= iAveragePixel:
                aAverage.append(1)
            else:
                aAverage.append(0)

    iDP = sum(aAverage) - len(aAverage) / 2
    if 0 < iDP:
        bExposure = True

    return bExposure

def VA(aSeparateImage, aBalacedImg, aObject, iHeightIndex, iWidthIndex):
    iHeight, iWidth = aSeparateImage.shape
    
    iMinHeight = iHeight
    iMaxHeight = -1
    iMinWidth = iWidth
    iMaxWidth = -1
    
    for iIndex in range(iHeightIndex, iHeight):
        if (0 == aSeparateImage[iIndex, iWidthIndex]):
            if iMinHeight > iIndex:
                iMinHeight = iIndex
            if iMaxHeight < iIndex:
                iMaxHeight = iIndex
        else:
            break

    for iIndex in range(iWidthIndex, iWidth):
        if (0 == aSeparateImage[iHeightIndex, iIndex]):
            if iMinWidth > iIndex:
                iMinWidth = iIndex
            if iMaxWidth < iIndex:
                iMaxWidth = iIndex
        else:
            break

    iCount = 0
    iSum = 0

    for iMarkedHeightIndex in range(iMinHeight, iMaxHeight + 1):
        for iMarkedWidthIndex in range(iMinWidth, iMaxWidth + 1):
            iCount += 1
            iSum += int(aBalacedImg[iMarkedHeightIndex, iMarkedWidthIndex])
            aSeparateImage[iMarkedHeightIndex, iMarkedWidthIndex] = 255

    iVA = 0
    iAverage = int(iSum / iCount)

    for iMarkedHeightIndex in range(iMinHeight, iMaxHeight + 1):
        for iMarkedWidthIndex in range(iMinWidth, iMaxWidth + 1):
            iVA += (int(aBalacedImg[iMarkedHeightIndex, iMarkedWidthIndex]) - iAverage) ** 2

    iVA = int(iVA / iCount)
    iArea = (iMaxHeight + 1 - iMinHeight) * (iMaxWidth + 1 - iMinWidth)

    aObject.append([iVA, iArea, iAverage, iMinHeight, iMaxHeight, iMinWidth, iMaxWidth])

    return aSeparateImage, aObject


def Separating(aSeparateImage, aBalacedImg):

    iHeight, iWidth = aSeparateImage.shape

    aObject = []

    for iHeightIndex in range(iHeight):
        for iWidthIndex in range(iWidth):
            if 0 == aSeparateImage[iHeightIndex, iWidthIndex]:
                aSeparateImage, aObject = VA(aSeparateImage, aBalacedImg, aObject, iHeightIndex, iWidthIndex)

    return aObject

def FinalBalance(aBalacedImg, aObject, bExposure):
    iAverageArea = 0
    iCountpx = 0
    aTextpx = []

    for iIndex in range(len(aObject)):
        iAverageArea += aObject[iIndex][1]

        for iIndexX in range(aObject[iIndex][3], aObject[iIndex][4] + 1):
            for iIndexY in range(aObject[iIndex][5], aObject[iIndex][6] + 1):
                iCountpx += 1
                aTextpx.append(aBalacedImg[iIndexX, iIndexY])

    aTextpx.sort()

    iAverageArea = int(iAverageArea / len(aObject))

    for iIndex in range(len(aObject)):
        if (1500 > aObject[iIndex][0] and aObject[iIndex][1] > (iAverageArea * 2)):
            for iIndexX in range(aObject[iIndex][3] + 2, aObject[iIndex][4] + 1):
                for iIndexY in range(aObject[iIndex][5] + 2, aObject[iIndex][6] + 1):
                    if (bExposure):
                        aBalacedImg[iIndexX, iIndexY] = aBalacedImg[iIndexX, iIndexY] - (aObject[iIndex][2] - 20) / 3
                    else:
                        aBalacedImg[iIndexX, iIndexY] = aBalacedImg[iIndexX, iIndexY] - (aObject[iIndex][2] - 20) / 2
        else:
            if (bExposure):
                iRP = 0
                iTT = 0

                iOneTenth = int(len(aTextpx) / 100)
                if (1 >iOneTenth):
                    iOneTenth = 1

                for iIndexText in range(iOneTenth):
                    iRP += aTextpx[iIndexText]

                iTwoThirds = int(len(aTextpx) * 2 / 3)
                if (1 > iTwoThirds):
                    iTwoThirds = 1

                for iIndexText in range(iTwoThirds):
                    iTT += aTextpx[iIndexText]

                iRP = int(iRP / iOneTenth)
                iTT = int(iTT / iTwoThirds)

                for iIndexX in range(aObject[iIndex][3], aObject[iIndex][4] + 1):
                    for iIndexY in range(aObject[iIndex][5], aObject[iIndex][6] + 1):
                        if (iTT > aBalacedImg[iIndexX, iIndexY]):
                            aBalacedImg[iIndexX, iIndexY] = iRP

    return aBalacedImg

def GrayImg(sImgName, iThreshold, iBalance):
    aImg = cv2.imread(sImgName, cv2.IMREAD_GRAYSCALE)

    aEdgeImg = EdgeImg(aImg)
    for iKernelIndex in range(len(aEdgeImg)):
        cv2.imwrite('Kernel_' + str(iKernelIndex + 1) + '.jpg', aEdgeImg[iKernelIndex])

    aEdgeAvgImg = EdgeAvgImg(aEdgeImg)
    cv2.imwrite('EdgeAvg.jpg', aEdgeAvgImg)
    del aEdgeImg

    aBinaryImg = BinaryImg(aEdgeAvgImg, iThreshold)
    cv2.imwrite('Binary.jpg', aBinaryImg)
    del aEdgeAvgImg

    aMarkedImg = MarkedImg(copy.deepcopy(aImg), aBinaryImg)
    cv2.imwrite('Marked.jpg', aMarkedImg)
    del aBinaryImg

    aLDIImg = LDIImg(copy.deepcopy(aMarkedImg), aMarkedImg)
    cv2.imwrite('LDI.jpg', aLDIImg)

    aBalacedImg = BalacedImg(copy.deepcopy(aImg), aLDIImg, aMarkedImg, iBalance)
    cv2.imwrite('Balaced.jpg', aBalacedImg)
    
    bExposure = ExposureDegree(aImg)
    
    aObject = Separating(copy.deepcopy(aMarkedImg), aBalacedImg)
    
    aFinalBalanceImg = FinalBalance(aBalacedImg, aObject, bExposure)
    cv2.imwrite('Final.jpg', aFinalBalanceImg)

def RGBImg(sImgName, iThreshold, iBalance):
    aRGBImg = cv2.imread(sImgName)
    
    aRGBFinalImg = np.zeros(aRGBImg.shape)

    for iIndex in range(3):
        aImg = aRGBImg[:, :, iIndex]

        aEdgeImg = EdgeImg(aImg)
        for iKernelIndex in range(len(aEdgeImg)):
            cv2.imwrite('Kernel_' + str(iKernelIndex + 1) + '.jpg', aEdgeImg[iKernelIndex])

        aEdgeAvgImg = EdgeAvgImg(aEdgeImg)
        cv2.imwrite('EdgeAvg.jpg', aEdgeAvgImg)
        del aEdgeImg

        aBinaryImg = BinaryImg(aEdgeAvgImg, iThreshold)
        cv2.imwrite('Binary.jpg', aBinaryImg)
        del aEdgeAvgImg

        aMarkedImg = MarkedImg(copy.deepcopy(aImg), aBinaryImg)
        cv2.imwrite('Marked.jpg', aMarkedImg)
        del aBinaryImg

        aLDIImg = LDIImg(copy.deepcopy(aMarkedImg), aMarkedImg)
        cv2.imwrite('LDI.jpg', aLDIImg)

        aBalacedImg = BalacedImg(copy.deepcopy(aImg), aLDIImg, aMarkedImg, iBalance)
        cv2.imwrite('Balaced.jpg', aBalacedImg)
        del aLDIImg
        
        bExposure = ExposureDegree(aImg)
        
        aObject = Separating(copy.deepcopy(aMarkedImg), aBalacedImg)
        del aMarkedImg
        
        aFinalBalanceImg = FinalBalance(aBalacedImg, aObject, bExposure)
        aRGBFinalImg[:, :, iIndex] = aFinalBalanceImg
        del aFinalBalanceImg, aBalacedImg, aObject

    cv2.imwrite('Final.jpg', aRGBFinalImg)

if __name__ == "__main__":
    iThreshold = 30
    iBalance = 260

    sImgName = input('Image Path:')
    bRGB = input('Gray/RGB(0/1):')
    
    if bRGB:
        RGBImg(sImgName, iThreshold, iBalance)
    else:
        GrayImg(sImgName, iThreshold, iBalance)
